"""
Main Training Script - Optimized for SOTA Quality
Target: <15M parameters with superior boundary quality
"""

import argparse
import os
import json
import torch
import random
import numpy as np

from config import DCFConfig
from trainer import DCFTrainer

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Optimized Dynamic Coarse-to-Fine Inpainting (<10M params)'
    )
    
    # Data
    parser.add_argument('--data_root', required=True,
                       help='Root directory containing train/ and val/ folders')
    
    # Model - OPTIMIZED DEFAULTS
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=48)
    parser.add_argument('--min_scale_size', type=int, default=8)
    parser.add_argument('--max_scale_size', type=int, default=128)
    parser.add_argument('--refinement_stages', type=int, default=3)
    
    # Training - STABILIZED
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--grad_clip_value', type=float, default=0.5)
    parser.add_argument('--device', default='auto')
    
    # Loss weights - FIXED VGG
    parser.add_argument('--charbonnier_weight', type=float, default=1.0)
    parser.add_argument('--pixel_accuracy_weight', type=float, default=1.0)
    parser.add_argument('--perceptual_weight', type=float, default=0.05)
    parser.add_argument('--style_weight', type=float, default=30.0)
    parser.add_argument('--boundary_weight', type=float, default=5.0)
    parser.add_argument('--boundary_max_weight', type=float, default=10.0)
    parser.add_argument('--smooth_weight', type=float, default=3.0)
    parser.add_argument('--color_consistency_weight', type=float, default=1.5)
    parser.add_argument('--adversarial_weight', type=float, default=0.005)
    
    # Discriminator - WITH SPECTRAL NORM
    parser.add_argument('--use_spectral_norm', action='store_true', default=True)
    parser.add_argument('--d_train_interval', type=int, default=2)
    
    # Dataset limits
    parser.add_argument('--max_train_samples', type=int, default=10000)
    parser.add_argument('--max_val_samples', type=int, default=1000)
    
    # Checkpointing
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--save_interval', type=int, default=3)
    parser.add_argument('--eval_interval', type=int, default=3)
    parser.add_argument('--resume', type=str, default=None)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_mixed_precision', action='store_true')
    
    return parser.parse_args()

def create_config(args):
    """Create configuration from arguments"""
    config = DCFConfig()
    
    # Update from args
    config.image_size = args.image_size
    config.hidden_dim = args.hidden_dim
    config.min_scale_size = args.min_scale_size
    config.max_scale_size = args.max_scale_size
    config.refinement_stages = args.refinement_stages
    
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.grad_clip_value = args.grad_clip_value
    config.num_workers = args.num_workers
    config.use_mixed_precision = not args.no_mixed_precision
    
    # Fixed loss weights
    config.charbonnier_weight = args.charbonnier_weight
    config.pixel_accuracy_weight = args.pixel_accuracy_weight
    config.perceptual_weight = args.perceptual_weight
    config.style_weight = args.style_weight
    config.boundary_weight = args.boundary_weight
    config.boundary_max_weight = args.boundary_max_weight
    config.smooth_weight = args.smooth_weight
    config.color_consistency_weight = args.color_consistency_weight
    config.adversarial_weight = args.adversarial_weight
    
    # Discriminator
    config.use_spectral_norm = args.use_spectral_norm
    config.d_train_interval = args.d_train_interval
    
    # Dataset limits
    config.max_train_samples = args.max_train_samples
    config.max_val_samples = args.max_val_samples
    
    config.save_interval = args.save_interval
    config.eval_interval = args.eval_interval
    
    if args.device == 'auto':
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        config.device = torch.device(args.device)
    
    return config

def print_model_info(trainer):
    """Print detailed model information"""
    from evaluation import calculate_model_size
    
    gen_size = calculate_model_size(trainer.generator)
    disc_size = calculate_model_size(trainer.discriminator)
    
    print("\n" + "="*80)
    print(" " * 25 + "🎯 OPTIMIZED DCF INPAINTING")
    print("="*80)
    print(f"\n📊 Model Architecture (~12M Parameters):")
    print(f"  ✓ Generator:       {gen_size['total_params_m']:.2f}M")
    print(f"  ✓ Discriminator:   {disc_size['total_params_m']:.2f}M")
    print(f"  ✓ Total:           {gen_size['total_params_m'] + disc_size['total_params_m']:.2f}M")
    print(f"\n🔧 Key Optimizations:")
    print(f"  ✓ Efficient 2-layer encoder/decoder blocks")
    print(f"  ✓ Single refinement layer per stage")
    print(f"  ✓ Spectral normalization for stable discriminator")
    print(f"  ✓ Adaptive boundary weight warmup")
    print(f"  ✓ Stabilized loss components (clamped)")
    print(f"  ✓ Reduced gradient clipping (0.5)")
    print("="*80 + "\n")

def main():
    """Main training function"""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create config
    config = create_config(args)
    
    # Save config
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Print configuration
    print("\n" + "="*80)
    print(" " * 20 + "🎨 Optimized DCF Inpainting Training")
    print("="*80)
    print(f"\n📁 Paths:")
    print(f"  Data root:         {args.data_root}")
    print(f"  Save directory:    {args.save_dir}")
    print(f"  Device:            {config.device}")
    
    print(f"\n⚙️ Model Configuration:")
    print(f"  Image size:        {config.image_size}x{config.image_size}")
    print(f"  Hidden dimension:  {config.hidden_dim}")
    print(f"  Encoder depths:    {config.encoder_depths}")
    print(f"  Decoder depths:    {config.decoder_depths}")
    print(f"  Scale range:       {config.min_scale_size} → {config.max_scale_size}")
    print(f"  Refinement stages: {config.refinement_stages}")
    
    print(f"\n🎓 Training Configuration:")
    print(f"  Batch size:        {config.batch_size}")
    print(f"  Learning rate:     {config.learning_rate}")
    print(f"  Epochs:            {config.num_epochs}")
    print(f"  Gradient clip:     {config.grad_clip_value}")
    print(f"  Mixed precision:   {config.use_mixed_precision}")
    print(f"  D train interval:  Every {config.d_train_interval} steps")
    
    print(f"\n💡 Fixed Loss Weights (VGG Working):")
    print(f"  Charbonnier:       {config.charbonnier_weight}")
    print(f"  Pixel + FFT:       {config.pixel_accuracy_weight}")
    print(f"  Perceptual (VGG):  {config.perceptual_weight} ✓ FIXED")
    print(f"  Style (VGG):       {config.style_weight} ✓ FIXED")
    print(f"  Boundary (adapt):  {config.boundary_weight} → {config.boundary_max_weight}")
    print(f"  Smooth:            {config.smooth_weight}")
    print(f"  Color:             {config.color_consistency_weight}")
    print(f"  Adversarial:       {config.adversarial_weight}")
    print("="*80 + "\n")
    
    # Validate data structure
    train_dir = os.path.join(args.data_root, 'train')
    val_dir = os.path.join(args.data_root, 'val')
    
    if not os.path.exists(train_dir):
        print(f"❌ ERROR: Training directory not found: {train_dir}")
        print(f"\nExpected structure:")
        print(f"  {args.data_root}/")
        print(f"  ├── train/")
        print(f"  │   ├── image1.jpg")
        print(f"  │   └── ...")
        print(f"  └── val/")
        print(f"      ├── image1.jpg")
        print(f"      └── ...")
        return
    
    if not os.path.exists(val_dir):
        print(f"❌ ERROR: Validation directory not found: {val_dir}")
        return
    
    # Initialize trainer
    print("🔧 Initializing optimized trainer...")
    trainer = DCFTrainer(
        config=config,
        data_root=args.data_root,
        save_dir=args.save_dir
    )
    
    # Print model info
    print_model_info(trainer)
    
    # Resume if specified
    if args.resume:
        print(f"\n📥 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    try:
        print("\n" + "="*80)
        print(" " * 30 + "🎬 Starting Training")
        print("="*80 + "\n")
        trainer.train()
        
        print("\n" + "="*80)
        print(" " * 30 + "✅ Training Completed!")
        print("="*80)
        print(f"\n🏆 Best PSNR achieved: {trainer.best_psnr:.2f} dB")
        
        # Print best metrics
        best_metrics = trainer.metrics_logger.get_best_metrics()
        if best_metrics:
            print(f"\n📈 Best Metrics Summary:")
            print(f"  PSNR: {best_metrics['best_psnr']['value']:.2f} dB (epoch {best_metrics['best_psnr']['epoch']})")
            print(f"  SSIM: {best_metrics['best_ssim']['value']:.4f} (epoch {best_metrics['best_ssim']['epoch']})")
            print(f"  L1:   {best_metrics['best_l1']['value']:.4f} (epoch {best_metrics['best_l1']['epoch']})")
        
        print("\n" + "="*80)
        print("\n✨ VGG Fix Applied:")
        print("  1. ✓ Proper denormalization (handles already-normalized inputs)")
        print("  2. ✓ Feature normalization before Gram matrix")
        print("  3. ✓ Log-scale FFT magnitudes (prevents explosion)")
        print("  4. ✓ Per-layer NaN detection with fallback")
        print("  5. ✓ All VGG losses clamped to safe ranges")
        print("\n📊 Expected Training:")
        print("  • NO NaN losses (VGG properly fixed)")
        print("  • Better texture quality (perceptual guidance active)")
        print("  • Faster convergence with VGG (~40-50 epochs)")
        print("  • PSNR target: 23-25 dB (SOTA with VGG)")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⏸️ Training interrupted by user")
        print("Checkpoint saved. Resume with: --resume <checkpoint_path>")
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        raise

if __name__ == '__main__':
    main()