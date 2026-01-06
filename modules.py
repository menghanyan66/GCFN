"""
Optimized Multi-Scale Feature Fusion Modules
IMPROVEMENTS: Single layer per stage, efficient attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicMaskAnalyzer(nn.Module):
    """Analyzes mask for dynamic scales"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def find_mask_center_and_bbox(self, mask):
        """Find mask center and bounding box"""
        B, _, H, W = mask.shape
        centers = []
        bbox_sizes = []
        bboxes = []
        
        for b in range(B):
            mask_b = mask[b, 0]
            coords = torch.nonzero(mask_b > 0.5, as_tuple=False)
            
            if coords.size(0) == 0:
                centers.append(torch.tensor([H//2, W//2], device=mask.device))
                bbox_sizes.append(torch.tensor(min(H, W) // 2, device=mask.device))
                bboxes.append(torch.tensor([H//4, W//4, 3*H//4, 3*W//4], device=mask.device))
            else:
                center_y = coords[:, 0].float().mean()
                center_x = coords[:, 1].float().mean()
                centers.append(torch.tensor([center_y, center_x], device=mask.device))
                
                y_min, y_max = coords[:, 0].min(), coords[:, 0].max()
                x_min, x_max = coords[:, 1].min(), coords[:, 1].max()
                
                height = y_max - y_min + 1
                width = x_max - x_min + 1
                size = max(height, width)
                bbox_sizes.append(size)
                
                cy = (y_min + y_max) // 2
                cx = (x_min + x_max) // 2
                half_size = size // 2
                
                y1 = max(0, cy - half_size)
                x1 = max(0, cx - half_size)
                y2 = min(H, cy + half_size)
                x2 = min(W, cx + half_size)
                
                bboxes.append(torch.tensor([y1, x1, y2, x2], device=mask.device))
        
        centers = torch.stack(centers)
        bbox_sizes = torch.stack(bbox_sizes)
        bboxes = torch.stack(bboxes)
        
        return centers, bbox_sizes, bboxes
    
    def generate_scales(self, bbox_size):
        """Generate dynamic scales"""
        scales = []
        current_size = min(bbox_size.item(), self.config.max_scale_size)
        
        while current_size >= self.config.min_scale_size:
            scales.append(int(current_size))
            current_size = int(current_size * self.config.scale_factor)
        
        if len(scales) == 0 or scales[-1] > self.config.min_scale_size:
            scales.append(self.config.min_scale_size)
            
        return scales
    
    def extract_quarter_patches(self, features, center, scale, image_size):
        """Extract 4 quarter patches"""
        B, C, H, W = features.shape
        cy, cx = center[0].item(), center[1].item()
        
        scale = min(scale, H, W)
        
        positions = [
            (max(0, int(cy - scale)), max(0, int(cx - scale))),
            (max(0, int(cy - scale)), min(W - scale, int(cx))),
            (min(H - scale, int(cy)), max(0, int(cx - scale))),
            (min(H - scale, int(cy)), min(W - scale, int(cx)))
        ]
        
        patches = []
        for y, x in positions:
            y_end = min(y + scale, H)
            x_end = min(x + scale, W)
            patch = features[:, :, y:y_end, x:x_end]
            
            if patch.shape[2] < scale or patch.shape[3] < scale:
                patch = F.pad(patch, (0, scale - patch.shape[3], 0, scale - patch.shape[2]))
            
            patches.append(patch)
        
        patches = torch.stack(patches, dim=1)
        return patches


class EfficientContextualAttentionFusion(nn.Module):
    """Efficient contextual attention with single post-processing layer"""
    def __init__(self, config, in_channels):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        
        reduction = 4
        reduced_channels = max(in_channels // reduction, 16)
        
        # Attention components
        self.query_conv = nn.Conv2d(in_channels, reduced_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, reduced_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # Single post-attention layer
        self.post_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, quarter_features, center_features, dropout_mask=None):
        B, num_quarters, C, H, W = quarter_features.shape
        
        quarter_flat = quarter_features.view(B * num_quarters, C, H, W)
        
        center_query = self.query_conv(center_features)
        center_query = center_query.view(B, -1, H * W).permute(0, 2, 1)
        
        quarter_keys = self.key_conv(quarter_flat)
        quarter_keys = quarter_keys.view(B, num_quarters, -1, H * W)
        
        quarter_values = self.value_conv(quarter_flat)
        quarter_values = quarter_values.view(B, num_quarters, C, H * W)
        
        attention_maps = []
        weighted_values = []
        
        for q in range(num_quarters):
            keys_q = quarter_keys[:, q]
            values_q = quarter_values[:, q]
            
            attention = torch.bmm(center_query, keys_q)
            attention = self.softmax(attention)
            
            if dropout_mask is not None:
                mask_q = dropout_mask[:, q].view(B, 1, 1)
                attention = attention * mask_q
            
            attention_maps.append(attention)
            out = torch.bmm(values_q, attention.permute(0, 2, 1))
            weighted_values.append(out)
        
        weighted_values = torch.stack(weighted_values, dim=1)
        aggregated = weighted_values.mean(dim=1)
        aggregated = aggregated.view(B, C, H, W)
        
        # Single post-processing layer
        aggregated = self.post_attention(aggregated)
        
        # Residual
        fused = self.gamma * aggregated + self.beta * center_features
        
        attention_maps = torch.stack(attention_maps, dim=1)
        
        return fused, attention_maps


class EnhancedProgressiveRefinementModule(nn.Module):
    """
    Progressive Refinement with single layer per stage
    OPTIMIZED: Reduced complexity while maintaining quality
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mask_analyzer = DynamicMaskAnalyzer(config)
        
        bottleneck_channels = config.encoder_depths[-1]
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(bottleneck_channels, config.hidden_dim, 1),
            nn.BatchNorm2d(config.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Fusion modules
        self.coarse_fusion = nn.ModuleList([
            EfficientContextualAttentionFusion(config, config.hidden_dim) 
            for _ in range(config.refinement_stages)
        ])
        
        # Single processing layer per stage
        self.stage_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
                nn.BatchNorm2d(config.hidden_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(config.refinement_stages)
        ])
        
        # Scale-specific refinement
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
                nn.BatchNorm2d(config.hidden_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(config.refinement_stages)
        ])
        
        self.dropout = nn.Dropout2d(p=config.dropout_rate)
        
    def compute_patch_importance(self, attention_maps, mask):
        """Compute importance scores"""
        B, num_quarters, HW_attn, _ = attention_maps.shape
        
        _, _, H_mask, W_mask = mask.shape
        HW_mask = H_mask * W_mask
        
        if HW_attn != HW_mask:
            H_attn = int(math.sqrt(HW_attn))
            W_attn = H_attn
            
            if H_attn * W_attn != HW_attn:
                for h in range(int(math.sqrt(HW_attn)), 0, -1):
                    if HW_attn % h == 0:
                        H_attn = h
                        W_attn = HW_attn // h
                        break
            
            mask_resized = F.interpolate(mask, size=(H_attn, W_attn), mode='nearest')
            mask_flat = mask_resized.view(B, -1)
        else:
            mask_flat = mask.view(B, -1)
        
        importance_scores = []
        for q in range(num_quarters):
            attention_q = attention_maps[:, q]
            mask_attention = torch.bmm(mask_flat.unsqueeze(1).float(), attention_q)
            score = mask_attention.sum(dim=-1)
            importance_scores.append(score)
        
        importance_scores = torch.cat(importance_scores, dim=1)
        threshold = torch.quantile(importance_scores, 1 - self.config.dropout_rate, dim=1, keepdim=True)
        dropout_mask = (importance_scores >= threshold).float()
        
        return dropout_mask
    
    def forward(self, features, mask):
        """Progressive refinement"""
        features = self.channel_adapter(features)
        centers, bbox_sizes, bboxes = self.mask_analyzer.find_mask_center_and_bbox(mask)
        
        stage_outputs = []
        current_features = features
        
        max_bbox_size = bbox_sizes.max()
        scales = self.mask_analyzer.generate_scales(max_bbox_size)
        
        if len(scales) == 0:
            scales = [self.config.min_scale_size]
        
        num_stages = min(len(scales), self.config.refinement_stages)
        
        for stage_idx in range(num_stages):
            scale = scales[stage_idx]
            
            # Extract patches
            quarter_patches = []
            for b in range(features.size(0)):
                patches = self.mask_analyzer.extract_quarter_patches(
                    current_features[b:b+1], centers[b], scale,
                    (features.size(2), features.size(3))
                )
                quarter_patches.append(patches)
            
            quarter_patches = torch.cat(quarter_patches, dim=0)
            
            # Resize
            target_size = current_features.shape[2:]
            quarter_resized = []
            for q in range(4):
                q_patch = quarter_patches[:, q]
                q_resized = F.interpolate(q_patch, size=target_size, 
                                        mode='bilinear', align_corners=False)
                quarter_resized.append(q_resized)
            quarter_resized = torch.stack(quarter_resized, dim=1)
            
            # Fusion
            fused, attention_maps = self.coarse_fusion[stage_idx](
                quarter_resized, current_features
            )
            
            # Importance dropout (except last stage)
            if stage_idx < num_stages - 1:
                dropout_mask = self.compute_patch_importance(attention_maps, mask)
                fused, _ = self.coarse_fusion[stage_idx](
                    quarter_resized, current_features, dropout_mask
                )
            
            # Single layer processing
            processed = self.stage_processors[stage_idx](fused)
            
            # Scale refinement
            if stage_idx < len(self.scale_convs):
                processed = self.scale_convs[stage_idx](processed)
            
            current_features = processed
            stage_outputs.append(processed)
        
        return current_features, stage_outputs


class EnhancedSmoothBlendingModule(nn.Module):
    """
    Enhanced Smooth Blending
    OPTIMIZED: Fewer scales, efficient processing
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Boundary detector
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(config.input_channels + 1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Multi-scale smoothing (3 scales)
        self.smooth_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(config.hidden_dim),
                nn.ReLU(inplace=True)
            )
            for k in [3, 5, 7]
        ])
        
        # Fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(config.hidden_dim * 3 + 1, config.hidden_dim * 2, 3, padding=1),
            nn.BatchNorm2d(config.hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.hidden_dim * 2, config.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(config.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Blend predictor
        self.blend_weight_predictor = nn.Sequential(
            nn.Conv2d(config.hidden_dim + 1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def create_smooth_mask(self, mask, kernel_size=11, sigma=2.5):
        """Create smoothly feathered mask"""
        device = mask.device
        
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        
        padding = kernel_size // 2
        smooth_mask = F.conv2d(mask.float(), kernel, padding=padding)
        
        return smooth_mask
    
    def extract_boundary_region(self, mask, dilation=7):
        """Extract boundary region"""
        padding = dilation // 2
        dilated = F.max_pool2d(mask, kernel_size=dilation, stride=1, padding=padding)
        eroded = -F.max_pool2d(-mask, kernel_size=dilation, stride=1, padding=padding)
        boundary = dilated - eroded
        return boundary
    
    def forward(self, features, input_image, mask):
        """Apply smooth blending"""
        # Boundary detection
        boundary_input = torch.cat([input_image, mask], dim=1)
        boundary_map = self.boundary_detector(boundary_input)
        
        # Smooth mask
        smooth_mask = self.create_smooth_mask(mask, 
                                              kernel_size=self.config.smooth_kernel_size,
                                              sigma=self.config.smooth_sigma)
        
        # Resize
        if smooth_mask.shape[2:] != features.shape[2:]:
            smooth_mask = F.interpolate(smooth_mask, size=features.shape[2:], 
                                       mode='bilinear', align_corners=False)
            boundary_map = F.interpolate(boundary_map, size=features.shape[2:],
                                        mode='bilinear', align_corners=False)
        
        # Multi-scale smoothing
        smoothed_features = []
        for smooth_layer in self.smooth_conv:
            smoothed = smooth_layer(features)
            smoothed_features.append(smoothed)
        
        # Fuse
        multi_scale = torch.cat(smoothed_features + [boundary_map], dim=1)
        fused = self.fusion_conv(multi_scale)
        
        # Blend weights
        blend_input = torch.cat([fused, smooth_mask], dim=1)
        blend_weights = self.blend_weight_predictor(blend_input)
        
        # Adaptive blending
        boundary_region = self.extract_boundary_region(mask, 
                                                       dilation=self.config.boundary_dilation)
        if boundary_region.shape[2:] != features.shape[2:]:
            boundary_region = F.interpolate(boundary_region, size=features.shape[2:],
                                          mode='bilinear', align_corners=False)
        
        adaptive_blend = blend_weights * boundary_region
        refined = fused * adaptive_blend + features * (1 - adaptive_blend)
        
        return refined, blend_weights