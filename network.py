"""
Optimized Network Architecture - ~12M parameters
IMPROVEMENTS: Spectral normalization, efficient blocks, better capacity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from modules import EnhancedProgressiveRefinementModule, EnhancedSmoothBlendingModule


class EfficientEncoderBlock(nn.Module):
    """Efficient Encoder with 2 conv layers + residual"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out, self.pool(out)


class EfficientDecoderBlock(nn.Module):
    """Efficient Decoder with 2 conv layers"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        
        self.conv1 = nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip):
        x = self.upconv(x)
        
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class OptimizedDCFInpaintingNetwork(nn.Module):
    """
    Optimized DCF Network - ~12M parameters
    - Efficient encoder/decoder (2 conv per block)
    - Strategic refinement integration
    - Balanced capacity
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(config.input_channels + 1, config.encoder_depths[0], 3, padding=1),
            nn.BatchNorm2d(config.encoder_depths[0]),
            nn.ReLU(inplace=True)
        )
        
        # EFFICIENT ENCODER
        self.encoder1 = EfficientEncoderBlock(config.encoder_depths[0], config.encoder_depths[0])
        self.encoder2 = EfficientEncoderBlock(config.encoder_depths[0], config.encoder_depths[1])
        self.encoder3 = EfficientEncoderBlock(config.encoder_depths[1], config.encoder_depths[2])
        self.encoder4 = EfficientEncoderBlock(config.encoder_depths[2], config.encoder_depths[3])
        
        # COMPACT BOTTLENECK
        self.bottleneck = nn.Sequential(
            nn.Conv2d(config.encoder_depths[3], config.encoder_depths[3], 3, padding=1),
            nn.BatchNorm2d(config.encoder_depths[3]),
            nn.ReLU(inplace=True)
        )
        
        # PROGRESSIVE REFINEMENT
        self.progressive_refinement = EnhancedProgressiveRefinementModule(config)
        
        # EFFICIENT DECODER
        self.decoder4 = EfficientDecoderBlock(
            config.decoder_depths[0], 
            config.encoder_depths[2] + config.hidden_dim, 
            config.decoder_depths[1]
        )
        self.decoder3 = EfficientDecoderBlock(
            config.decoder_depths[1], 
            config.encoder_depths[1], 
            config.decoder_depths[2]
        )
        self.decoder2 = EfficientDecoderBlock(
            config.decoder_depths[2], 
            config.encoder_depths[0], 
            config.decoder_depths[3]
        )
        self.decoder1 = EfficientDecoderBlock(
            config.decoder_depths[3], 
            config.encoder_depths[0], 
            config.decoder_depths[3]
        )
        
        # SMOOTH BLENDING
        self.smooth_blending = EnhancedSmoothBlendingModule(config)
        
        # Output layers
        self.output_conv = nn.Sequential(
            nn.Conv2d(config.decoder_depths[3], config.decoder_depths[3] // 2, 3, padding=1),
            nn.BatchNorm2d(config.decoder_depths[3] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.decoder_depths[3] // 2, config.input_channels, 3, padding=1),
            nn.Tanh()
        )
        
        # Stage outputs
        self.stage_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config.hidden_dim, config.input_channels, 3, padding=1),
                nn.Tanh()
            )
            for _ in range(config.refinement_stages)
        ])
        
    def forward(self, masked_image, mask, return_stages=False):
        # Encoder
        x = torch.cat([masked_image, mask], dim=1)
        x = self.input_conv(x)
        
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Progressive refinement
        refined_features, stage_features = self.progressive_refinement(x, mask)
        
        # Resize refined features
        if refined_features.shape[2:] != skip3.shape[2:]:
            refined_features = F.interpolate(
                refined_features, 
                size=skip3.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Combine with skip
        enhanced_skip3 = torch.cat([skip3, refined_features], dim=1)
        
        # Decoder
        x = self.decoder4(x, enhanced_skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder2(x, skip1)
        x = self.decoder1(x, skip1)
        
        # Smooth blending
        x, blend_map = self.smooth_blending(x, masked_image, mask)
        
        # Output
        output = self.output_conv(x)
        
        # Combine with original
        final_output = masked_image * (1 - mask) + output * mask
        
        result = {
            'output': final_output,
            'raw_output': output,
            'blend_map': blend_map
        }
        
        # Stage outputs
        if return_stages:
            stage_outputs = []
            for i, stage_feat in enumerate(stage_features):
                if stage_feat.shape[2:] != masked_image.shape[2:]:
                    stage_feat = F.interpolate(
                        stage_feat,
                        size=masked_image.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                if i < len(self.stage_outputs):
                    stage_out = self.stage_outputs[i](stage_feat)
                    stage_out = masked_image * (1 - mask) + stage_out * mask
                    stage_outputs.append(stage_out)
            
            result['stage_outputs'] = stage_outputs
        
        return result


class SpectralNormDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with Spectral Normalization
    KEY FIX: Prevents discriminator from becoming too strong
    """
    def __init__(self, config):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True, use_sn=True):
            """Discriminator block with optional spectral norm"""
            conv = nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)
            if use_sn and config.use_spectral_norm:
                conv = spectral_norm(conv)
            
            layers = [conv]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # Build discriminator with spectral norm
        self.model = nn.Sequential(
            *discriminator_block(config.input_channels, config.discriminator_filters, 
                               normalize=False, use_sn=True),
            *discriminator_block(config.discriminator_filters, config.discriminator_filters * 2,
                               use_sn=True),
            *discriminator_block(config.discriminator_filters * 2, config.discriminator_filters * 4,
                               use_sn=True),
            nn.Conv2d(config.discriminator_filters * 4, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)