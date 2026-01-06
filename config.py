"""
Optimized Configuration for SOTA Inpainting
Target: <15M parameters with superior quality
"""

import torch

class DCFConfig:
    def __init__(self):
        # ===== OPTIMIZED MODEL ARCHITECTURE (~12M params) =====
        self.input_channels = 3
        self.hidden_dim = 48  # Reduced from 64, balanced capacity
        
        # Efficient encoder/decoder depths
        self.encoder_depths = [48, 96, 192, 384]  # Optimized progression
        self.decoder_depths = [384, 192, 96, 48]
        
        # ===== DYNAMIC MULTI-SCALE SETTINGS =====
        self.min_scale_size = 8   # Balanced detail vs computation
        self.max_scale_size = 128
        self.scale_factor = 0.5
        self.num_quarters = 4
        
        # ===== PROGRESSIVE REFINEMENT =====
        self.refinement_stages = 3  # Optimal for 128x128
        self.dropout_rate = 0.15    # Reduced for better learning
        self.attention_heads = 4    # Efficient attention
        self.refinement_layers_per_stage = 1  # Single layer per stage
        
        # ===== TRAINING PARAMETERS =====
        self.batch_size = 32
        self.learning_rate = 0.0002  # Increased for faster convergence
        self.num_epochs = 50
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ===== MEMORY OPTIMIZATION =====
        self.use_mixed_precision = True
        self.gradient_checkpointing = False
        self.pin_memory = True if torch.cuda.is_available() else False
        self.num_workers = 4
        
        # ===== FIXED LOSS WEIGHTS (VGG WORKING NOW) =====
        # Reconstruction
        self.charbonnier_weight = 1.0
        self.charbonnier_eps = 1e-3
        
        # Pixel accuracy with safe FFT
        self.pixel_accuracy_weight = 1.0
        
        # FIXED: Perceptual & Style with proper denormalization
        self.perceptual_weight = 0.05   # Conservative start
        self.style_weight = 30.0        # Conservative start (was 50.0)
        
        # Adversarial
        self.adversarial_weight = 0.005
        
        # Boundary & Smoothness - ADAPTIVE WEIGHTING
        self.boundary_weight = 5.0      # Start value
        self.smooth_weight = 3.0
        self.color_consistency_weight = 1.5
        
        # Progressive supervision - BALANCED
        self.progressive_weights = [0.8, 0.5, 0.3]
        
        # Adaptive boundary settings
        self.use_adaptive_boundary = True
        self.boundary_warmup_epochs = 10
        self.boundary_max_weight = 10.0
        
        # Smoothness Module Settings
        self.smooth_kernel_size = 11  # Reduced from 15
        self.smooth_sigma = 2.5       # Reduced from 3.0
        self.boundary_dilation = 7    # Reduced from 9
        
        # ===== DATA PARAMETERS =====
        self.image_size = 128
        self.mask_ratio_range = (0.05, 0.15)
        
        # ===== DISCRIMINATOR - WITH SPECTRAL NORM =====
        self.discriminator_layers = 3
        self.discriminator_filters = 48  # Reduced from 64
        self.use_spectral_norm = True    # NEW: Prevent D overfitting
        
        # Discriminator training schedule
        self.d_train_interval = 2  # Train D every N steps
        self.g_train_interval = 1  # Train G every step
        
        # ===== GRADIENT MANAGEMENT =====
        self.grad_clip_value = 0.5  # REDUCED from 1.0 for stability
        self.grad_accumulation_steps = 1
        
        # ===== TRAINING SETTINGS =====
        self.save_interval = 3
        self.eval_interval = 3
        self.log_interval = 20
        
        # Dataset limits
        self.max_train_samples = 8000
        self.max_val_samples = 800
        
    def to_dict(self):
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        for key, value in config_dict.items():
            if key == 'device':
                setattr(config, key, torch.device(value))
            else:
                setattr(config, key, value)
        return config
    
    def get_adaptive_boundary_weight(self, epoch):
        """Compute adaptive boundary weight based on training progress"""
        if not self.use_adaptive_boundary:
            return self.boundary_weight
        
        if epoch < self.boundary_warmup_epochs:
            # Gradual warmup
            progress = epoch / self.boundary_warmup_epochs
            return self.boundary_weight + (self.boundary_max_weight - self.boundary_weight) * progress
        else:
            return self.boundary_max_weight