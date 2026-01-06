"""
Stabilized Training Pipeline
FIXES: Numerical stability, adaptive boundary, balanced D/G training
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import os
import logging
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils

from network import OptimizedDCFInpaintingNetwork, SpectralNormDiscriminator
from loss import StabilizedDCFInpaintingLoss
from dataset import create_dataloaders
from evaluation import InpaintingEvaluator, MetricsLogger, calculate_model_size


class DCFTrainer:
    """Stabilized Trainer with adaptive boundary weighting"""
    
    def __init__(self, config, data_root, save_dir='./checkpoints'):
        self.config = config
        self.save_dir = save_dir
        self.current_epoch = 0
        self.global_step = 0
        self.best_psnr = 0.0
        
        # Setup logging
        self._setup_logging()
        
        # Create directories
        self.checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        self.logs_dir = os.path.join(save_dir, 'logs')
        self.samples_dir = os.path.join(save_dir, 'samples')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        
        # Initialize optimized models
        self.logger.info("Initializing optimized models (~12M params)...")
        self.generator = OptimizedDCFInpaintingNetwork(config).to(config.device)
        self.discriminator = SpectralNormDiscriminator(config).to(config.device)
        
        # Print model info
        gen_size = calculate_model_size(self.generator)
        disc_size = calculate_model_size(self.discriminator)
        self.logger.info(f"✓ Generator: {gen_size['total_params_m']:.2f}M parameters")
        self.logger.info(f"✓ Discriminator: {disc_size['total_params_m']:.2f}M parameters")
        self.logger.info(f"✓ Total: {gen_size['total_params_m'] + disc_size['total_params_m']:.2f}M parameters")
        
        # Initialize loss with FIXED VGG
        self.criterion = StabilizedDCFInpaintingLoss(config).to(config.device)
        
        # Move FIXED VGG to device
        for extractor in self.criterion.perceptual_loss.feature_extractor:
            extractor.to(config.device)
        for extractor in self.criterion.style_loss.feature_extractor:
            extractor.to(config.device)
        
        # Optimizers with improved settings
        self.optimizer_G = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(0.5, 0.999),
            weight_decay=1e-4
        )
        
        # Slower discriminator learning rate
        self.optimizer_D = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.learning_rate * 0.25,  # REDUCED from 0.5
            betas=(0.5, 0.999),
            weight_decay=1e-4
        )
        
        # Cosine annealing schedulers
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_G, T_max=config.num_epochs, eta_min=1e-6
        )
        
        self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_D, T_max=config.num_epochs, eta_min=1e-7
        )
        
        # Mixed precision
        self.scaler_G = GradScaler() if config.use_mixed_precision else None
        self.scaler_D = GradScaler() if config.use_mixed_precision else None
        
        # Data loaders
        self.logger.info("Creating data loaders...")
        self.train_loader, self.val_loader = create_dataloaders(config, data_root)
        
        # Evaluator
        self.evaluator = InpaintingEvaluator(config)
        self.metrics_logger = MetricsLogger()
        
        # Training stats
        self.train_losses = {
            'total_g': [], 'rec': [], 'pixel': [], 'perceptual': [],
            'style': [], 'boundary': [], 'smooth': [],
            'color': [], 'progressive': [],
            'adversarial': [], 'discriminator': []
        }
        
        self.logger.info("=" * 70)
        self.logger.info("FIXED LOSS CONFIGURATION (VGG WORKING):")
        self.logger.info(f"  ✓ FIXED VGG denormalization (proper handling of normalized inputs)")
        self.logger.info(f"  ✓ Safe Gram matrix with feature normalization")
        self.logger.info(f"  ✓ Log-scale FFT magnitudes for stability")
        self.logger.info(f"  ✓ Adaptive Boundary: {config.boundary_weight} → {config.boundary_max_weight}")
        self.logger.info(f"  ✓ Perceptual weight: {config.perceptual_weight}")
        self.logger.info(f"  ✓ Style weight: {config.style_weight}")
        self.logger.info("=" * 70)
    
    def _setup_logging(self):
        """Setup logging"""
        os.makedirs(self.save_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.save_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DCFTrainer')
    
    def train_discriminator_step(self, batch):
        """Stabilized discriminator training"""
        self.optimizer_D.zero_grad()
        
        masked_image = batch['masked_image'].to(self.config.device)
        mask = batch['mask'].to(self.config.device)
        original = batch['original_image'].to(self.config.device)
        
        if self.config.use_mixed_precision:
            with autocast():
                with torch.no_grad():
                    fake_outputs = self.generator(masked_image, mask, return_stages=False)
                    fake_images = fake_outputs['output']
                
                real_pred = self.discriminator(original)
                fake_pred = self.discriminator(fake_images.detach())
                
                d_loss_dict = self.criterion.discriminator_loss(real_pred, fake_pred)
                d_loss = d_loss_dict['discriminator_loss']
            
            self.scaler_D.scale(d_loss).backward()
            self.scaler_D.unscale_(self.optimizer_D)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 
                                          max_norm=self.config.grad_clip_value)
            self.scaler_D.step(self.optimizer_D)
            self.scaler_D.update()
        else:
            with torch.no_grad():
                fake_outputs = self.generator(masked_image, mask, return_stages=False)
                fake_images = fake_outputs['output']
            
            real_pred = self.discriminator(original)
            fake_pred = self.discriminator(fake_images.detach())
            
            d_loss_dict = self.criterion.discriminator_loss(real_pred, fake_pred)
            d_loss = d_loss_dict['discriminator_loss']
            
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 
                                          max_norm=self.config.grad_clip_value)
            self.optimizer_D.step()
        
        return {
            'discriminator_loss': d_loss.item(),
            'real_loss': d_loss_dict['real_loss'].item(),
            'fake_loss': d_loss_dict['fake_loss'].item()
        }
    
    def train_generator_step(self, batch):
        """Stabilized generator training"""
        self.optimizer_G.zero_grad()
        
        masked_image = batch['masked_image'].to(self.config.device)
        mask = batch['mask'].to(self.config.device)
        original = batch['original_image'].to(self.config.device)
        
        if self.config.use_mixed_precision:
            with autocast():
                outputs = self.generator(masked_image, mask, return_stages=True)
                generated = outputs['output']
                
                fake_pred = self.discriminator(generated)
                real_pred = self.discriminator(original)
                
                loss_dict = self.criterion(
                    outputs=outputs,
                    target=original,
                    mask=mask,
                    discriminator_real=real_pred,
                    discriminator_fake=fake_pred
                )
                
                total_loss = loss_dict['total_loss']
            
            # Check for NaN/Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self.logger.warning(f"⚠ Invalid loss detected: {total_loss.item()}, skipping step")
                return {k: 0.0 for k in ['total_loss', 'rec_loss', 'pixel_loss',
                                         'perceptual_loss', 'style_loss', 'boundary_loss',
                                         'smooth_loss', 'color_loss', 'progressive_loss',
                                         'adversarial_loss']}
            
            self.scaler_G.scale(total_loss).backward()
            self.scaler_G.unscale_(self.optimizer_G)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 
                                          max_norm=self.config.grad_clip_value)
            self.scaler_G.step(self.optimizer_G)
            self.scaler_G.update()
        else:
            outputs = self.generator(masked_image, mask, return_stages=True)
            generated = outputs['output']
            
            fake_pred = self.discriminator(generated)
            real_pred = self.discriminator(original)
            
            loss_dict = self.criterion(
                outputs=outputs,
                target=original,
                mask=mask,
                discriminator_real=real_pred,
                discriminator_fake=fake_pred
            )
            
            total_loss = loss_dict['total_loss']
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self.logger.warning(f"⚠ Invalid loss detected, skipping step")
                return {k: 0.0 for k in ['total_loss', 'rec_loss', 'pixel_loss',
                                         'perceptual_loss', 'style_loss', 'boundary_loss',
                                         'smooth_loss', 'color_loss', 'progressive_loss',
                                         'adversarial_loss']}
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 
                                          max_norm=self.config.grad_clip_value)
            self.optimizer_G.step()
        
        return {
            'total_loss': loss_dict['total_loss'].item(),
            'rec_loss': loss_dict['rec_loss'].item(),
            'pixel_loss': loss_dict['pixel_loss'].item(),
            'perceptual_loss': loss_dict['perceptual_loss'].item(),
            'style_loss': loss_dict['style_loss'].item(),
            'boundary_loss': loss_dict['boundary_loss'].item(),
            'smooth_loss': loss_dict['smooth_loss'].item(),
            'color_loss': loss_dict['color_loss'].item(),
            'progressive_loss': loss_dict['progressive_loss'].item() if isinstance(loss_dict['progressive_loss'], (int, float)) else loss_dict['progressive_loss'].item(),
            'adversarial_loss': loss_dict['adversarial_loss'].item() if isinstance(loss_dict['adversarial_loss'], (int, float)) else loss_dict['adversarial_loss'].item()
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch with adaptive boundary weight"""
        self.generator.train()
        self.discriminator.train()
        
        # Update adaptive boundary weight
        self.criterion.update_boundary_weight(epoch)
        current_bw = self.criterion.current_boundary_weight
        
        epoch_losses = {
            'total_g': [], 'rec': [], 'pixel': [], 'perceptual': [],
            'style': [], 'boundary': [], 'smooth': [],
            'color': [], 'progressive': [],
            'adversarial': [], 'discriminator': []
        }
        
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{self.config.num_epochs} (BW={current_bw:.1f})',
            ncols=160,
            leave=True
        )
        
        for batch_idx, batch in enumerate(pbar):
            self.global_step += 1
            
            # Train discriminator at interval
            if batch_idx % self.config.d_train_interval == 0:
                d_losses = self.train_discriminator_step(batch)
            else:
                d_losses = {'discriminator_loss': 0.0, 'real_loss': 0.0, 'fake_loss': 0.0}
            
            # Train generator
            g_losses = self.train_generator_step(batch)
            
            # Update stats (only valid losses)
            if g_losses['total_loss'] > 0:
                for key in ['total_g', 'rec', 'pixel', 'perceptual', 'style', 
                           'boundary', 'smooth', 'color', 'progressive', 'adversarial']:
                    if key == 'total_g':
                        epoch_losses[key].append(g_losses['total_loss'])
                    else:
                        epoch_losses[key].append(g_losses[f'{key}_loss'])
            
            if d_losses['discriminator_loss'] > 0:
                epoch_losses['discriminator'].append(d_losses['discriminator_loss'])
            
            # Update progress
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'G': f"{g_losses['total_loss']:.2f}",
                    'D': f"{d_losses['discriminator_loss']:.3f}",
                    'Bnd': f"{g_losses['boundary_loss']:.2f}",
                    'Best': f"{self.best_psnr:.1f}dB"
                })
            
            # Memory cleanup
            if batch_idx % 50 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()
        
        pbar.close()
        
        # Calculate averages
        avg_losses = {}
        for k, vals in epoch_losses.items():
            valid_vals = [v for v in vals if v > 0 and not np.isnan(v) and not np.isinf(v)]
            avg_losses[k] = np.mean(valid_vals) if valid_vals else 0.0
        
        # Update schedulers
        self.scheduler_G.step()
        self.scheduler_D.step()
        
        # Logging
        self.logger.info(
            f"Epoch {epoch} | "
            f"G: {avg_losses['total_g']:.3f} | "
            f"D: {avg_losses['discriminator']:.3f} | "
            f"Boundary: {avg_losses['boundary']:.3f} (weight={current_bw:.1f}) | "
            f"LR: {self.scheduler_G.get_last_lr()[0]:.6f}"
        )
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.generator.eval()
        
        all_generated = []
        all_targets = []
        max_samples = 200
        
        pbar = tqdm(self.val_loader, desc='Validation', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            if len(all_generated) * self.config.batch_size >= max_samples:
                break
            
            masked_image = batch['masked_image'].to(self.config.device)
            mask = batch['mask'].to(self.config.device)
            original = batch['original_image'].to(self.config.device)
            
            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.generator(masked_image, mask, return_stages=False)
            else:
                outputs = self.generator(masked_image, mask, return_stages=False)
            
            generated = outputs['output']
            
            all_generated.append(generated.cpu())
            all_targets.append(original.cpu())
            
            # Save samples
            if batch_idx == 0:
                self._save_sample_images(masked_image, mask, generated, original, 
                                        outputs.get('blend_map'), epoch)
        
        all_generated = torch.cat(all_generated, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics = self.evaluator.evaluate(all_generated, all_targets)
        
        self.logger.info(
            f"📊 Validation | "
            f"PSNR: {metrics['psnr']:.2f}±{metrics['psnr_std']:.2f} | "
            f"SSIM: {metrics['ssim']:.4f}±{metrics['ssim_std']:.4f} | "
            f"L1: {metrics['l1_error']:.4f}"
        )
        
        self.metrics_logger.update(metrics, epoch)
        self.metrics_logger.plot_metrics(
            save_path=os.path.join(self.logs_dir, 'metrics.png')
        )
        
        return metrics
    
    def _save_sample_images(self, masked, mask, generated, original, blend_map, epoch):
        """Save sample visualizations"""
        def denormalize(img):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
            return torch.clamp(img * std + mean, 0, 1)
        
        masked_d = denormalize(masked[:8])
        generated_d = denormalize(generated[:8])
        original_d = denormalize(original[:8])
        
        comparison = torch.cat([masked_d, generated_d, original_d], dim=0)
        grid = vutils.make_grid(comparison, nrow=8, padding=2)
        
        save_path = os.path.join(self.samples_dir, f'epoch_{epoch:03d}.png')
        vutils.save_image(grid, save_path)
    
    def save_checkpoint(self, epoch, metrics=None, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'config': self.config.to_dict(),
            'best_psnr': self.best_psnr,
            'metrics': metrics,
            'train_losses': self.train_losses
        }
        
        if self.scaler_G:
            checkpoint['scaler_G'] = self.scaler_G.state_dict()
            checkpoint['scaler_D'] = self.scaler_D.state_dict()
        
        ckpt_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
        torch.save(checkpoint, ckpt_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"🏆 Best model saved: PSNR={self.best_psnr:.2f}")
        
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        self._cleanup_old_checkpoints(keep_last=3)
    
    def _cleanup_old_checkpoints(self, keep_last=3):
        """Remove old checkpoints"""
        checkpoints = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('checkpoint_epoch_')
        ])
        
        if len(checkpoints) > keep_last:
            for ckpt in checkpoints[:-keep_last]:
                os.remove(os.path.join(self.checkpoint_dir, ckpt))
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        self.logger.info(f"Loading: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_psnr = checkpoint.get('best_psnr', 0.0)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        if 'scheduler_G_state_dict' in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        
        if 'scaler_G' in checkpoint and self.scaler_G:
            self.scaler_G.load_state_dict(checkpoint['scaler_G'])
            self.scaler_D.load_state_dict(checkpoint['scaler_D'])
    
    def train(self):
        """Main training loop"""
        self.logger.info("=" * 70)
        self.logger.info("🚀 Starting Optimized DCF Training")
        self.logger.info("=" * 70)
        
        start_epoch = self.current_epoch + 1
        
        try:
            for epoch in range(start_epoch, self.config.num_epochs + 1):
                self.current_epoch = epoch
                
                # Train
                train_losses = self.train_epoch(epoch)
                
                for key, value in train_losses.items():
                    self.train_losses[key].append(value)
                
                # Validate
                if epoch % self.config.eval_interval == 0:
                    metrics = self.validate(epoch)
                    
                    is_best = metrics['psnr'] > self.best_psnr
                    if is_best:
                        self.best_psnr = metrics['psnr']
                    
                    self.save_checkpoint(epoch, metrics, is_best)
                else:
                    if epoch % self.config.save_interval == 0:
                        self.save_checkpoint(epoch)
            
            self.logger.info("=" * 70)
            self.logger.info(f"✅ Training completed! Best PSNR: {self.best_psnr:.2f}")
            self.logger.info("=" * 70)
            
        except KeyboardInterrupt:
            self.logger.info("\n⏸ Training interrupted")
            self.save_checkpoint(self.current_epoch)
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}", exc_info=True)
            self.save_checkpoint(self.current_epoch)
            raise