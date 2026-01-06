"""
Inference Script for Dynamic Coarse-to-Fine Inpainting
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import argparse
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from network import DCFInpaintingNetwork
from config import DCFConfig

class DCFInference:
    """Inference class for DCF Inpainting"""
    
    def __init__(self, checkpoint_path, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        # FIX 1: Set weights_only=False to allow loading complex checkpoint data
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load config
        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            self.config = DCFConfig.from_dict(checkpoint['config'])
        else:
            self.config = checkpoint['config']
        
        self.config.device = device
        
        # Initialize model
        self.model = DCFInpaintingNetwork(self.config).to(device)
        self.model.load_state_dict(checkpoint['generator_state_dict'])
        self.model.eval()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {device}")
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor, original_size
    
    def preprocess_mask(self, mask_path, target_size=None):
        """Load and preprocess mask"""
        if mask_path is None:
            # Create center mask
            h, w = self.config.image_size, self.config.image_size
            mask = torch.zeros(1, 1, h, w)
            center_h, center_w = h // 4, w // 4
            mask[:, :, center_h:center_h*3, center_w:center_w*3] = 1
            return mask
        
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.config.image_size, self.config.image_size))
        mask_tensor = transforms.ToTensor()(mask).unsqueeze(0)
        mask_tensor = (mask_tensor > 0.5).float()
        return mask_tensor
    
    def denormalize(self, tensor):
        """Denormalize tensor"""
        # FIX 2: Use (3, 1, 1) to broadcast correctly without adding a 4th dimension
        # Also ensure mean/std are on the same device as the input tensor
        mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device).view(3, 1, 1)
        return torch.clamp(tensor * std + mean, 0, 1)
    
    def inpaint(self, image_path, mask_path=None, save_path=None):
        """
        Perform inpainting
        
        Args:
            image_path: Path to input image
            mask_path: Path to mask (None for automatic center mask)
            save_path: Path to save result
        """
        # Preprocess
        image_tensor, original_size = self.preprocess_image(image_path)
        mask_tensor = self.preprocess_mask(mask_path)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)
        
        # Apply mask
        masked_image = image_tensor * (1 - mask_tensor)
        
        # Inference
        print("Running inpainting...")
        with torch.no_grad():
            outputs = self.model(masked_image, mask_tensor, return_stages=False)
        
        # Post-process
        result = outputs['output'].cpu()
        result_denorm = self.denormalize(result.squeeze(0))
        
        # Resize to original size if needed
        if original_size != (self.config.image_size, self.config.image_size):
            result_pil = transforms.ToPILImage()(result_denorm)
            result_pil = result_pil.resize(original_size, Image.LANCZOS)
            result_denorm = transforms.ToTensor()(result_pil)
        
        # Save
        if save_path:
            save_image(result_denorm, save_path)
            print(f"Result saved to {save_path}")
        
        return result_denorm
    
    def batch_inpaint(self, input_dir, output_dir, mask_dir=None):
        """
        Batch inpainting
        
        Args:
            input_dir: Directory with input images
            output_dir: Directory to save results
            mask_dir: Directory with masks (optional)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all images
        image_files = [f for f in os.listdir(input_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {len(image_files)} images...")
        
        for img_file in image_files:
            print(f"\nProcessing {img_file}...")
            
            img_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, f"inpainted_{img_file}")
            
            # Get mask if available
            mask_path = None
            if mask_dir:
                base_name = os.path.splitext(img_file)[0]
                mask_name = f"{base_name}_mask.png"
                potential_mask = os.path.join(mask_dir, mask_name)
                if os.path.exists(potential_mask):
                    mask_path = potential_mask
            
            try:
                self.inpaint(img_path, mask_path, output_path)
                print(f"Ã¢Å“â€œ Successfully processed {img_file}")
            except Exception as e:
                print(f"Ã¢Å“â€” Error processing {img_file}: {e}")
                continue
        
        print(f"\nÃ¢Å“â€œ Batch processing complete! Results in {output_dir}")
    
    def create_comparison(self, image_path, mask_path=None, save_path=None):
        """Create comparison visualization"""
        # Get result
        result = self.inpaint(image_path, mask_path)
        
        # Load original
        image_tensor, _ = self.preprocess_image(image_path)
        mask_tensor = self.preprocess_mask(mask_path)
        masked_image = image_tensor * (1 - mask_tensor)
        
        # Denormalize
        original = self.denormalize(image_tensor.squeeze(0))
        masked = self.denormalize(masked_image.squeeze(0))
        mask_vis = mask_tensor.squeeze(0).repeat(3, 1, 1)
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        def to_pil(t):
            return transforms.ToPILImage()(t)
        
        axes[0].imshow(to_pil(original))
        axes[0].set_title('Original', fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(to_pil(mask_vis), cmap='gray')
        axes[1].set_title('Mask', fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(to_pil(masked))
        axes[2].set_title('Masked Input', fontweight='bold')
        axes[2].axis('off')
        
        axes[3].imshow(to_pil(result))
        axes[3].set_title('Inpainted Result', fontweight='bold')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to {save_path}")
        
        plt.show()
        return fig


def main():
    parser = argparse.ArgumentParser(description='DCF Inpainting Inference')
    
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--input', help='Input image path')
    parser.add_argument('--mask', help='Mask image path (optional)')
    parser.add_argument('--output', help='Output path')
    parser.add_argument('--comparison', help='Save comparison visualization')
    
    # Batch mode
    parser.add_argument('--batch_mode', action='store_true')
    parser.add_argument('--input_dir', help='Input directory for batch mode')
    parser.add_argument('--output_dir', help='Output directory for batch mode')
    parser.add_argument('--mask_dir', help='Mask directory for batch mode')
    
    parser.add_argument('--device', default='auto')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Initialize inference
    inferencer = DCFInference(args.checkpoint, device)
    
    if args.batch_mode:
        # Batch processing
        if not args.input_dir or not args.output_dir:
            print("Error: --input_dir and --output_dir required for batch mode")
            return
        
        inferencer.batch_inpaint(
            args.input_dir,
            args.output_dir,
            args.mask_dir
        )
    else:
        # Single image
        if not args.input or not args.output:
            print("Error: --input and --output required for single image mode")
            return
        
        # Run inpainting
        inferencer.inpaint(args.input, args.mask, args.output)
        
        # Create comparison if requested
        if args.comparison:
            inferencer.create_comparison(args.input, args.mask, args.comparison)

if __name__ == '__main__':
    main()