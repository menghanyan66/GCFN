"""
Evaluation Metrics for Inpainting
"""

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt

class InpaintingEvaluator:
    """Evaluator for inpainting quality"""
    def __init__(self, config):
        self.config = config
    
    def evaluate(self, generated_images, target_images):
        """
        Evaluate generated images
        
        Args:
            generated_images: [N, 3, H, W]
            target_images: [N, 3, H, W]
        Returns:
            dict of metrics
        """
        # Convert to numpy
        gen_np = self.tensor_to_numpy(generated_images)
        target_np = self.tensor_to_numpy(target_images)
        
        # Calculate PSNR
        psnr_values = []
        for i in tqdm(range(len(gen_np)), desc='Computing PSNR', leave=False):
            psnr = peak_signal_noise_ratio(target_np[i], gen_np[i], data_range=1.0)
            psnr_values.append(psnr)
        
        # Calculate SSIM
        ssim_values = []
        for i in tqdm(range(len(gen_np)), desc='Computing SSIM', leave=False):
            ssim = structural_similarity(
                target_np[i], gen_np[i],
                channel_axis=2,
                data_range=1.0
            )
            ssim_values.append(ssim)
        
        # Calculate L1 error
        l1_error = torch.abs(generated_images - target_images).mean().item()
        
        metrics = {
            'psnr': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'ssim': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values),
            'l1_error': l1_error
        }
        
        return metrics
    
    def tensor_to_numpy(self, tensor):
        """Convert normalized tensor to numpy [0, 1]"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        # To numpy
        numpy_array = tensor.cpu().numpy().transpose(0, 2, 3, 1)
        return numpy_array


class MetricsLogger:
    """Logger for tracking metrics during training"""
    def __init__(self):
        self.metrics_history = {
            'psnr': [],
            'ssim': [],
            'l1_error': [],
            'epochs': []
        }
    
    def update(self, metrics, epoch):
        """Update metrics history"""
        self.metrics_history['psnr'].append(metrics['psnr'])
        self.metrics_history['ssim'].append(metrics['ssim'])
        self.metrics_history['l1_error'].append(metrics['l1_error'])
        self.metrics_history['epochs'].append(epoch)
    
    def get_best_metrics(self):
        """Get best achieved metrics"""
        if not self.metrics_history['epochs']:
            return None
        
        best_psnr_idx = np.argmax(self.metrics_history['psnr'])
        best_ssim_idx = np.argmax(self.metrics_history['ssim'])
        best_l1_idx = np.argmin(self.metrics_history['l1_error'])
        
        return {
            'best_psnr': {
                'value': self.metrics_history['psnr'][best_psnr_idx],
                'epoch': self.metrics_history['epochs'][best_psnr_idx]
            },
            'best_ssim': {
                'value': self.metrics_history['ssim'][best_ssim_idx],
                'epoch': self.metrics_history['epochs'][best_ssim_idx]
            },
            'best_l1': {
                'value': self.metrics_history['l1_error'][best_l1_idx],
                'epoch': self.metrics_history['epochs'][best_l1_idx]
            }
        }
    
    def plot_metrics(self, save_path=None):
        """Plot metrics evolution"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # PSNR
        axes[0].plot(self.metrics_history['epochs'], self.metrics_history['psnr'], 'b-o')
        axes[0].set_title('PSNR', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].grid(True, alpha=0.3)
        
        # SSIM
        axes[1].plot(self.metrics_history['epochs'], self.metrics_history['ssim'], 'g-s')
        axes[1].set_title('SSIM', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('SSIM')
        axes[1].grid(True, alpha=0.3)
        
        # L1 Error
        axes[2].plot(self.metrics_history['epochs'], self.metrics_history['l1_error'], 'r-^')
        axes[2].set_title('L1 Error', fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('L1 Error')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to {save_path}")
        
        plt.close()
        return fig


def calculate_model_size(model):
    """Calculate model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_m': total_params / 1e6,
        'trainable_params_m': trainable_params / 1e6
    }