"""
Stabilized Loss Functions with Adaptive Boundary Weighting
FIXES: Numerical stability, boundary quality, balanced training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss for stable reconstruction"""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)


class EnhancedPixelAccuracyLoss(nn.Module):
    """
    Enhanced Pixel Loss with SAFE frequency matching
    FIXED: Proper FFT magnitude normalization
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, mask=None):
        """Compute pixel-level accuracy with safe frequency matching"""
        # Standard L1 loss
        if mask is not None:
            pred_masked = pred * mask
            target_masked = target * mask
            spatial_loss = F.l1_loss(pred_masked, target_masked)
        else:
            spatial_loss = F.l1_loss(pred, target)
        
        # SAFE Frequency domain matching
        try:
            # Apply FFT
            pred_fft = torch.fft.rfft2(pred, norm='ortho')
            target_fft = torch.fft.rfft2(target, norm='ortho')
            
            # Get magnitudes
            pred_mag = torch.abs(pred_fft)
            target_mag = torch.abs(target_fft)
            
            # CRITICAL: Normalize magnitudes to prevent explosion
            # Use log-scale for stability
            pred_mag_log = torch.log1p(pred_mag)  # log(1 + x) is more stable
            target_mag_log = torch.log1p(target_mag)
            
            # Compute loss on log-normalized magnitudes
            freq_loss = F.l1_loss(pred_mag_log, target_mag_log)
            
            # Safety check
            if torch.isnan(freq_loss) or torch.isinf(freq_loss):
                freq_loss = torch.tensor(0.0, device=pred.device)
            else:
                freq_loss = torch.clamp(freq_loss, max=2.0)
            
        except Exception as e:
            # If FFT fails for any reason, just use spatial loss
            freq_loss = torch.tensor(0.0, device=pred.device)
        
        # Combine with small weight on frequency component
        total_loss = spatial_loss + 0.1 * freq_loss
        
        return total_loss


class AdaptiveBoundaryLoss(nn.Module):
    """
    Adaptive Boundary Loss with gradient continuity
    KEY FIX: Distance-weighted smoothing for better transitions
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_gradient(self, img):
        """Compute image gradients"""
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return gx, gy

    def extract_boundary_region(self, mask):
        """Extract boundary with distance weighting"""
        padding = self.kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size=self.kernel_size, 
                               stride=1, padding=padding)
        eroded = -F.max_pool2d(-mask, kernel_size=self.kernel_size, 
                               stride=1, padding=padding)
        boundary = dilated - eroded
        
        # Distance-based weighting (closer to boundary = higher weight)
        distance_from_mask = F.max_pool2d(mask, kernel_size=self.kernel_size,
                                          stride=1, padding=padding) - mask
        weight_map = torch.exp(-distance_from_mask * 3)  # Exponential decay
        
        return boundary, weight_map

    def forward(self, pred, target, mask):
        """Compute adaptive boundary loss"""
        boundary, weight_map = self.extract_boundary_region(mask)
        
        pred_gx, pred_gy = self.compute_gradient(pred)
        target_gx, target_gy = self.compute_gradient(target)
        
        # Weighted gradient matching
        loss_x = torch.abs((pred_gx - target_gx) * boundary * weight_map)
        loss_y = torch.abs((pred_gy - target_gy) * boundary * weight_map)
        
        boundary_area = (boundary * weight_map).sum() + 1e-8
        total_loss = (loss_x.sum() + loss_y.sum()) / boundary_area
        
        # STABILIZED: Clamp to prevent explosion
        total_loss = torch.clamp(total_loss, max=5.0)
        
        return total_loss


class SmoothTransitionLoss(nn.Module):
    """Enhanced Smooth Transition with local statistics matching"""
    def __init__(self, kernel_size=11, sigma=2.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel())
    
    def _create_gaussian_kernel(self):
        ax = torch.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * self.sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)
    
    def compute_local_stats(self, img, mask):
        """Compute local mean and variance"""
        B, C, H, W = img.shape
        padding = self.kernel_size // 2
        
        # Local mean
        img_smooth = F.conv2d(
            img.view(B*C, 1, H, W),
            self.gaussian_kernel,
            padding=padding
        ).view(B, C, H, W)
        
        # Local variance
        diff_sq = (img - img_smooth) ** 2
        local_var = F.conv2d(
            diff_sq.view(B*C, 1, H, W),
            self.gaussian_kernel,
            padding=padding
        ).view(B, C, H, W)
        
        return img_smooth, local_var
    
    def forward(self, pred, target, mask):
        """Compute smooth transition loss"""
        padding = self.kernel_size // 2
        boundary_mask = F.max_pool2d(mask, kernel_size=self.kernel_size,
                                     stride=1, padding=padding) - mask
        
        # Match local statistics at boundary
        pred_mean, pred_var = self.compute_local_stats(pred, boundary_mask)
        target_mean, target_var = self.compute_local_stats(target, boundary_mask)
        
        mean_loss = F.l1_loss(pred_mean * boundary_mask, target_mean * boundary_mask)
        var_loss = F.l1_loss(pred_var * boundary_mask, target_var * boundary_mask)
        
        # STABILIZED: Clamp
        total_loss = torch.clamp(mean_loss + var_loss * 0.5, max=3.0)
        
        return total_loss


class ColorConsistencyLoss(nn.Module):
    """Stabilized Color Consistency"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
    
    def extract_boundary_region(self, mask):
        padding = self.kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size=self.kernel_size,
                               stride=1, padding=padding)
        eroded = -F.max_pool2d(-mask, kernel_size=self.kernel_size,
                               stride=1, padding=padding)
        return dilated - eroded
    
    def compute_color_statistics(self, img, region_mask):
        B, C, H, W = img.shape
        mask_expanded = region_mask.expand(B, C, H, W)
        
        masked_sum = (img * mask_expanded).sum(dim=[2, 3])
        mask_count = mask_expanded.sum(dim=[2, 3]) + 1e-8
        mean = masked_sum / mask_count
        
        diff = img - mean.view(B, C, 1, 1)
        masked_var = ((diff ** 2) * mask_expanded).sum(dim=[2, 3]) / mask_count
        std = torch.sqrt(masked_var + 1e-8)
        
        return mean, std
    
    def forward(self, pred, target, mask):
        boundary = self.extract_boundary_region(mask)
        
        pred_mean, pred_std = self.compute_color_statistics(pred, boundary)
        target_mean, target_std = self.compute_color_statistics(target, boundary)
        
        mean_loss = F.mse_loss(pred_mean, target_mean)
        std_loss = F.mse_loss(pred_std, target_std)
        
        # STABILIZED
        total_loss = torch.clamp(mean_loss + std_loss, max=2.0)
        
        return total_loss


class PerceptualLoss(nn.Module):
    """FIXED VGG Perceptual Loss - handles already-normalized inputs"""
    def __init__(self, layers=[3, 8, 15]):
        super().__init__()
        vgg = models.vgg19(weights='IMAGENET1K_V1').features
        self.feature_extractor = nn.ModuleList()
        
        prev_layer = 0
        for layer_idx in layers:
            self.feature_extractor.append(vgg[prev_layer:layer_idx+1])
            prev_layer = layer_idx + 1
            
        for extractor in self.feature_extractor:
            extractor.eval()
            for param in extractor.parameters():
                param.requires_grad = False
        
        # ImageNet normalization (for reference, but NOT used since input is pre-normalized)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def denormalize_to_01(self, x):
        """
        Convert from ImageNet-normalized space to [0,1] for VGG
        Input x is already normalized with mean/std, we need to reverse it
        """
        # x is in normalized space: (img - mean) / std
        # To get back to [0,1]: x * std + mean
        denorm = x * self.std + self.mean
        # Clamp to valid range and ensure no NaN
        denorm = torch.clamp(denorm, 0.0, 1.0)
        # Safety check for NaN/Inf
        if torch.isnan(denorm).any() or torch.isinf(denorm).any():
            print("⚠ NaN/Inf detected in denormalization, using safe fallback")
            return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)  # Simple rescale as fallback
        return denorm

    def forward(self, pred, target):
        """Forward pass with proper denormalization"""
        # Convert from normalized space to [0,1] for VGG
        pred_01 = self.denormalize_to_01(pred.float())
        target_01 = self.denormalize_to_01(target.float())
        
        # Safety check
        if torch.isnan(pred_01).any() or torch.isnan(target_01).any():
            print("⚠ NaN in VGG input, returning zero loss")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        loss = 0.0
        pred_feat = pred_01
        target_feat = target_01
        
        for idx, extractor in enumerate(self.feature_extractor):
            pred_feat = extractor(pred_feat)
            target_feat = extractor(target_feat)
            
            # Safety check at each layer
            if torch.isnan(pred_feat).any() or torch.isnan(target_feat).any():
                print(f"⚠ NaN in VGG layer {idx}, stopping feature extraction")
                break
            
            # Normalize features to prevent explosion
            pred_feat_norm = pred_feat / (pred_feat.norm(dim=1, keepdim=True) + 1e-8)
            target_feat_norm = target_feat / (target_feat.norm(dim=1, keepdim=True) + 1e-8)
            
            layer_loss = F.l1_loss(pred_feat_norm, target_feat_norm)
            loss += layer_loss
        
        # STABILIZED with reasonable upper bound
        loss = torch.clamp(loss, max=3.0)
        
        return loss


class StyleLoss(nn.Module):
    """FIXED Style Loss with safe Gram matrix computation"""
    def __init__(self, layers=[3, 8, 15]):
        super().__init__()
        vgg = models.vgg19(weights='IMAGENET1K_V1').features
        self.feature_extractor = nn.ModuleList()
        prev_layer = 0
        for layer_idx in layers:
            self.feature_extractor.append(vgg[prev_layer:layer_idx+1])
            prev_layer = layer_idx + 1
        
        for extractor in self.feature_extractor:
            extractor.eval()
            for param in extractor.parameters():
                param.requires_grad = False
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def denormalize_to_01(self, x):
        """Safe denormalization to [0,1]"""
        denorm = x * self.std + self.mean
        denorm = torch.clamp(denorm, 0.0, 1.0)
        if torch.isnan(denorm).any() or torch.isinf(denorm).any():
            return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)
        return denorm

    def gram_matrix(self, features):
        """Safe Gram matrix computation with normalization"""
        B, C, H, W = features.shape
        
        # Normalize features first to prevent explosion
        features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        
        features_flat = features_norm.view(B, C, H * W)
        
        # Compute gram matrix
        gram = torch.bmm(features_flat, features_flat.transpose(1, 2))
        
        # Normalize by spatial dimensions
        gram = gram / (C * H * W + 1e-8)
        
        # Safety check
        if torch.isnan(gram).any() or torch.isinf(gram).any():
            print("⚠ NaN/Inf in Gram matrix, returning zero")
            return torch.zeros_like(gram)
        
        return gram

    def forward(self, pred, target):
        """Forward with safe Gram computation"""
        pred_01 = self.denormalize_to_01(pred.float())
        target_01 = self.denormalize_to_01(target.float())
        
        if torch.isnan(pred_01).any() or torch.isnan(target_01).any():
            print("⚠ NaN in Style input, returning zero loss")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        loss = 0.0
        p_feat, t_feat = pred_01, target_01
        
        for idx, extractor in enumerate(self.feature_extractor):
            p_feat = extractor(p_feat)
            t_feat = extractor(t_feat)
            
            if torch.isnan(p_feat).any() or torch.isnan(t_feat).any():
                print(f"⚠ NaN in Style layer {idx}, stopping")
                break
            
            p_gram = self.gram_matrix(p_feat)
            t_gram = self.gram_matrix(t_feat)
            
            layer_loss = F.l1_loss(p_gram, t_gram)
            loss += layer_loss
        
        # STABILIZED
        loss = torch.clamp(loss, max=2.0)
        
        return loss


class AdversarialLoss(nn.Module):
    """Stabilized LSGAN Loss"""
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, pred, target_is_real):
        pred = pred.float()
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        loss = self.loss(pred, target)
        # STABILIZED
        return torch.clamp(loss, max=2.0)


class StabilizedDCFInpaintingLoss(nn.Module):
    """
    FIXED Loss - Proper VGG handling with denormalization
    KEY FIX: VGG losses now handle already-normalized inputs correctly
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Core losses
        self.reconstruction_loss = CharbonnierLoss(eps=config.charbonnier_eps)
        self.pixel_accuracy_loss = EnhancedPixelAccuracyLoss()  # Fixed FFT
        
        # FIXED: VGG losses with proper denormalization
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        
        # Smoothness losses (STABLE)
        self.boundary_loss = AdaptiveBoundaryLoss(
            kernel_size=config.boundary_dilation
        )
        self.smooth_loss = SmoothTransitionLoss(
            kernel_size=config.smooth_kernel_size,
            sigma=config.smooth_sigma
        )
        self.color_consistency_loss = ColorConsistencyLoss(
            kernel_size=config.boundary_dilation
        )
        
        # Realism
        self.adversarial_loss = AdversarialLoss()
        
        # Track current boundary weight
        self.current_boundary_weight = config.boundary_weight
    
    def update_boundary_weight(self, epoch):
        """Update adaptive boundary weight"""
        self.current_boundary_weight = self.config.get_adaptive_boundary_weight(epoch)
    
    def forward(self, outputs, target, mask, discriminator_real=None, 
                discriminator_fake=None):
        """Compute loss with FIXED VGG handling"""
        generated = outputs['output']
        
        # Ensure float32
        gen_f32 = generated.float()
        target_f32 = target.float()
        mask_f32 = mask.float()
        
        # 1. Reconstruction (Charbonnier)
        rec_loss = self.reconstruction_loss(gen_f32, target_f32)
        
        # 2. FIXED: Pixel accuracy with safe FFT
        pixel_loss = self.pixel_accuracy_loss(generated, target, mask)
        
        # 3. FIXED: Perceptual & Style with proper denormalization
        try:
            perc_loss = self.perceptual_loss(generated, target)
            if torch.isnan(perc_loss) or torch.isinf(perc_loss):
                perc_loss = torch.tensor(0.0, device=generated.device, requires_grad=True)
                print("⚠ Perceptual loss produced NaN, using 0")
        except Exception as e:
            perc_loss = torch.tensor(0.0, device=generated.device, requires_grad=True)
            print(f"⚠ Perceptual loss failed: {e}")
        
        try:
            style_loss = self.style_loss(generated, target)
            if torch.isnan(style_loss) or torch.isinf(style_loss):
                style_loss = torch.tensor(0.0, device=generated.device, requires_grad=True)
                print("⚠ Style loss produced NaN, using 0")
        except Exception as e:
            style_loss = torch.tensor(0.0, device=generated.device, requires_grad=True)
            print(f"⚠ Style loss failed: {e}")
        
        # 4. ADAPTIVE Boundary Loss
        boundary_loss = self.boundary_loss(generated, target, mask)
        
        # 5. Smooth Transitions
        smooth_loss = self.smooth_loss(generated, target, mask)
        
        # 6. Color Consistency
        color_loss = self.color_consistency_loss(generated, target, mask)
        
        # 7. Progressive Supervision
        prog_loss = torch.tensor(0.0, device=generated.device)
        if 'stage_outputs' in outputs and outputs['stage_outputs']:
            for idx, stage_out in enumerate(outputs['stage_outputs']):
                weight = self.config.progressive_weights[idx] if idx < len(self.config.progressive_weights) else 0.1
                prog_loss += weight * self.reconstruction_loss(stage_out.float(), target_f32)
        
        # 8. Adversarial
        adv_loss = torch.tensor(0.0, device=generated.device)
        if discriminator_fake is not None:
            adv_loss = self.adversarial_loss(discriminator_fake, target_is_real=True)
        
        # TOTAL LOSS with FIXED VGG
        total_loss = (
            self.config.charbonnier_weight * rec_loss +
            self.config.pixel_accuracy_weight * pixel_loss +
            self.config.perceptual_weight * perc_loss +
            self.config.style_weight * style_loss +
            self.current_boundary_weight * boundary_loss +
            self.config.smooth_weight * smooth_loss +
            self.config.color_consistency_weight * color_loss +
            0.2 * prog_loss +
            self.config.adversarial_weight * adv_loss
        )
        
        # CRITICAL: Clamp total loss
        total_loss = torch.clamp(total_loss, max=100.0)
        
        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("⚠ Total loss is NaN/Inf - component analysis:")
            print(f"  rec: {rec_loss.item()}")
            print(f"  pixel: {pixel_loss.item()}")
            print(f"  perc: {perc_loss.item()}")
            print(f"  style: {style_loss.item()}")
            print(f"  boundary: {boundary_loss.item()}")
            print(f"  smooth: {smooth_loss.item()}")
            print(f"  color: {color_loss.item()}")
            # Return a safe fallback loss
            total_loss = rec_loss + boundary_loss  # Use only stable components
        
        return {
            'total_loss': total_loss,
            'rec_loss': rec_loss,
            'pixel_loss': pixel_loss,
            'perceptual_loss': perc_loss,
            'style_loss': style_loss,
            'boundary_loss': boundary_loss,
            'smooth_loss': smooth_loss,
            'color_loss': color_loss,
            'progressive_loss': prog_loss,
            'adversarial_loss': adv_loss
        }

    def discriminator_loss(self, real_pred, fake_pred):
        """Stabilized discriminator loss"""
        real_loss = self.adversarial_loss(real_pred, target_is_real=True)
        fake_loss = self.adversarial_loss(fake_pred, target_is_real=False)
        return {
            'discriminator_loss': (real_loss + fake_loss) * 0.5,
            'real_loss': real_loss,
            'fake_loss': fake_loss
        }