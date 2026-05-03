# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 22:45:50 2025

@author: fbtek
"""

import torch
import torch.nn as nn
import numpy as np
import math
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from AdaptiveLocal2DLayerv2 import AdaptiveLocal2DLayer  # <-- Replace with actual path


def make_alc2d(input_size, output_size):
    return AdaptiveLocal2DLayer(input_size, output_size,
                                           channel_separate=False,
                                           normed=True, layer_norm=False,
                                           n_embedding=None,
                                           si_init=(0.03,0.03),
                                           shared_weights=False, activ='relu')

class ViTWithALC(nn.Module):
    """
    Vision Transformer with AdaptiveLocalLayer2D
    Extracts patch tokens, reconstructs them into 2D feature map, applies ALC, then linear classifier
    """
    def __init__(self, base_vit, num_classes, patch_size=None, img_size=224, alc_output=(32,32)):
        super().__init__()
        
        # Store ViT components
        self.patch_embed = base_vit.conv_proj  # Patch embedding layer
        self.pos_embed = base_vit.encoder.pos_embedding  # Positional embedding
        self.encoder = base_vit.encoder.layers  # Transformer encoder layers
        self.layer_norm = base_vit.encoder.ln  # Final layer norm
        
        # Get patch and image info
        self.patch_size = patch_size or base_vit.patch_size
        self.img_size = img_size
        self.num_patches = (img_size // self.patch_size) ** 2
        self.patches_per_side = img_size // self.patch_size
        
        # Get embedding dimension
        self.embed_dim = base_vit.hidden_dim
        
        print(f"ViT Config:")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Image size: {img_size}")
        print(f"  Patches per side: {self.patches_per_side}")
        print(f"  Total patches: {self.num_patches}")
        print(f"  Embedding dim: {self.embed_dim}")
        print(f"  Reconstructed feature map: {self.embed_dim}x{self.patches_per_side}x{self.patches_per_side}")
        
        # AdaptiveLocalLayer2D
     
        feature_map_shape = (self.embed_dim, self.patches_per_side, self.patches_per_side)
        self.alc = make_alc2d(feature_map_shape, output_size=alc_output)
        
        # Final classifier
        self.fc = nn.Linear(np.prod(alc_output), num_classes)
        
        # Dropout (if needed)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        B = x.shape[0]
        
        # 1. Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)  # (B, embed_dim, H_patches, W_patches)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # 2. Add positional embedding (includes CLS token)
        # We need to handle CLS token separately
        cls_token = self.pos_embed[:, :1, :]  # CLS token embedding
        pos_embed_patches = self.pos_embed[:, 1:, :]  # Patch position embeddings
        
        # Add CLS token to sequence
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # 3. Pass through transformer encoder layers
        for layer in self.encoder:
            x = layer(x)
        
        # 4. Apply layer norm
        x = self.layer_norm(x)
        
        # 5. Remove CLS token and keep only patch tokens
        patch_tokens = x[:, 1:, :]  # (B, num_patches, embed_dim)
        
        # 6. Reconstruct 2D feature map from patches
        feature_map = self.reconstruct_feature_map(patch_tokens)  # (B, embed_dim, H_patches, W_patches)
        
        # 7. Apply AdaptiveLocalLayer2D
        x = self.alc(feature_map)  # (B, alc_output[0] * alc_output[1])
        
        # 8. Flatten and apply final classifier
        x = torch.flatten(x, 1)
        #x = self.dropout(x)
        x = self.fc(x)
        
        return x

    def reconstruct_feature_map(self, patch_tokens):
        """
        Reconstruct 2D feature map from patch tokens
        Args:
            patch_tokens: (B, num_patches, embed_dim)
        Returns:
            feature_map: (B, embed_dim, patches_per_side, patches_per_side)
        """
        B, num_patches, embed_dim = patch_tokens.shape
        
        # Reshape to 2D grid
        feature_map = patch_tokens.view(B, self.patches_per_side, self.patches_per_side, embed_dim)
        
        # Permute to (B, embed_dim, H, W) format
        feature_map = feature_map.permute(0, 3, 1, 2)
        
        return feature_map


class ViTStandardHead(nn.Module):
    """
    Standard Vision Transformer classifier using only the CLS token.
    Equivalent to ViTWithALC, but no 2D reconstruction or ALC2D.
    """
    def __init__(self, base_vit, num_classes):
        super().__init__()
        
        self.patch_embed = base_vit.conv_proj
        self.pos_embed = base_vit.encoder.pos_embedding
        self.encoder = base_vit.encoder.layers
        self.layer_norm = base_vit.encoder.ln

        self.embed_dim = base_vit.hidden_dim
        #self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)                        # (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)               # (B, num_patches, embed_dim)

        # Add CLS token and positional embedding
        cls_token = self.pos_embed[:, :1, :]           # (1, 1, embed_dim)
        cls_tokens = cls_token.expand(B, -1, -1)       # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)          # (B, num_patches + 1, embed_dim)
        x = x + self.pos_embed                         # (B, num_patches + 1, embed_dim)

        # Transformer
        for layer in self.encoder:
            x = layer(x)

        # Layer norm
        x = self.layer_norm(x)

        # Classification from CLS token
        cls_token_final = x[:, 0, :]                   # (B, embed_dim)
        #x = self.dropout(cls_token_final)
        return self.fc(cls_token_final)


class ViTWithALC_MultiScale(nn.Module):
    """
    ViT with ALC that can extract features from multiple transformer layers
    """
    def __init__(self, base_vit, num_classes, extract_layer=-1, patch_size=None, 
                 img_size=224, alc_output=(32,32)):
        super().__init__()
        
        self.patch_embed = base_vit.conv_proj
        self.pos_embed = base_vit.encoder.pos_embedding
        self.encoder_layers = base_vit.encoder.layers
        self.layer_norm = base_vit.encoder.ln
        
        # Configuration
        self.patch_size = patch_size or base_vit.patch_size
        self.img_size = img_size
        self.num_patches = (img_size // self.patch_size) ** 2
        self.patches_per_side = img_size // self.patch_size
        self.embed_dim = base_vit.hidden_dim
        self.extract_layer = extract_layer  # -1 for last layer, -2 for second to last, etc.
        
        print(f"ViT MultiScale Config:")
        print(f"  Extracting from layer: {extract_layer} (total layers: {len(self.encoder_layers)})")
        print(f"  Feature map: {self.embed_dim}x{self.patches_per_side}x{self.patches_per_side}")
        
        # AdaptiveLocalLayer2D

        feature_map_shape = (self.embed_dim, self.patches_per_side, self.patches_per_side)
        self.alc = make_alc2d(feature_map_shape, output_size=alc_output)
        
        # Final classifier
        self.fc = nn.Linear(np.prod(alc_output), num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add positional embedding
        cls_tokens = self.pos_embed[:, :1, :].expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        # Pass through transformer layers up to extract_layer
        target_layer = len(self.encoder_layers) + self.extract_layer if self.extract_layer < 0 else self.extract_layer
        
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i == target_layer:
                break
        
        # Apply layer norm only if we're at the last layer
        if target_layer == len(self.encoder_layers) - 1:
            x = self.layer_norm(x)
        
        # Remove CLS token
        patch_tokens = x[:, 1:, :]
        
        # Reconstruct and process
        feature_map = self.reconstruct_feature_map(patch_tokens)
        x = self.alc(feature_map)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def reconstruct_feature_map(self, patch_tokens):
        B, num_patches, embed_dim = patch_tokens.shape
        feature_map = patch_tokens.view(B, self.patches_per_side, self.patches_per_side, embed_dim)
        feature_map = feature_map.permute(0, 3, 1, 2)
        return feature_map

class ViTWithALC_HighRes(nn.Module):
    """
    ViT with ALC that upsamples the reconstructed feature map for higher resolution
    """
    def __init__(self, base_vit, num_classes, upsample_factor=2, patch_size=None, 
                 img_size=224, alc_output=(32,32)):
        super().__init__()
        
        self.patch_embed = base_vit.conv_proj
        self.pos_embed = base_vit.encoder.pos_embedding
        self.encoder = base_vit.encoder.layers
        self.layer_norm = base_vit.encoder.ln
        
        # Configuration
        self.patch_size = patch_size or base_vit.patch_size
        self.img_size = img_size
        self.num_patches = (img_size // self.patch_size) ** 2
        self.patches_per_side = img_size // self.patch_size
        self.embed_dim = base_vit.hidden_dim
        self.upsample_factor = upsample_factor
        
        # Upsampled dimensions
        self.upsampled_size = self.patches_per_side * upsample_factor
        
        print(f"ViT HighRes Config:")
        print(f"  Original patches: {self.patches_per_side}x{self.patches_per_side}")
        print(f"  Upsampled size: {self.upsampled_size}x{self.upsampled_size}")
        print(f"  Feature channels: {self.embed_dim}")
        
        # Upsampling layer
        self.upsample = nn.Upsample(
            size=(self.upsampled_size, self.upsampled_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Optional: Convolutional layer to refine upsampled features
        self.refine_conv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim, 3, padding=1),
        )
        
        # AdaptiveLocalLayer2D

        feature_map_shape = (self.embed_dim, self.upsampled_size, self.upsampled_size)
        self.alc = make_alc2d(feature_map_shape, output_size=alc_output)
        
        # Final classifier
        self.fc = nn.Linear(np.prod(alc_output), num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Standard ViT processing
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.pos_embed[:, :1, :].expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for layer in self.encoder:
            x = layer(x)
        
        x = self.layer_norm(x)
        patch_tokens = x[:, 1:, :]
        
        # Reconstruct feature map
        feature_map = self.reconstruct_feature_map(patch_tokens)
        
        # Upsample for higher resolution
        feature_map = self.upsample(feature_map)
        
        # Optional refinement
        feature_map = self.refine_conv(feature_map)
        
        # Apply ALC and classifier
        x = self.alc(feature_map)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def reconstruct_feature_map(self, patch_tokens):
        B, num_patches, embed_dim = patch_tokens.shape
        feature_map = patch_tokens.view(B, self.patches_per_side, self.patches_per_side, embed_dim)
        feature_map = feature_map.permute(0, 3, 1, 2)
        return feature_map

def create_vit_with_alc(model_name="vit_b_16", variant="standard", num_classes=1000, 
                        w_weights="DEFAULT",  alc_output=(32,32), **kwargs):
    """
    Create ViT model with AdaptiveLocalLayer2D
    
    Args:
        model_name: "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"
        variant: "standard", "multiscale", "highres"
        num_classes: number of output classes
        **kwargs: additional arguments for specific variants
    """
    
    # Load pretrained ViT
    if model_name == "vit_b_16":
        base_vit = vit_b_16(weights=w_weights)
        patch_size = 16
    elif model_name == "vit_b_32":
        base_vit = vit_b_32(weights=w_weights)
        patch_size = 32
    elif model_name == "vit_l_16":
        base_vit = vit_l_16(weights=w_weights)
        patch_size = 16
    elif model_name == "vit_l_32":
        base_vit = vit_l_32(weights=w_weights)
        patch_size = 32
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create model variant
    if variant == "standard":
        model = ViTWithALC(base_vit, num_classes, patch_size=patch_size, alc_output=alc_output, **kwargs)
    elif variant == "multiscale":
        extract_layer = kwargs.get("extract_layer", -1)
        model = ViTWithALC_MultiScale(base_vit, num_classes, extract_layer=extract_layer, 
                                    patch_size=patch_size, alc_output=alc_output, **kwargs)
    elif variant == "highres":
        upsample_factor = kwargs.get("upsample_factor", 2)
        model = ViTWithALC_HighRes(base_vit, num_classes, upsample_factor=upsample_factor,
                                 patch_size=patch_size, alc_output=alc_output, **kwargs)
    elif variant =="baseline":
        model =  ViTStandardHead(base_vit, num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    return model

# Example usage and testing
if __name__ == "__main__":
    # Placeholder for AdaptiveLocalLayer2D
    
    
    print("Testing different ViT configurations:")
    
    # Test different models
    models_to_test = [
        ("vit_b_16", "standard"),
        ("vit_b_32", "standard"), 
        ("vit_b_16", "multiscale"),
        ("vit_b_16", "highres"),
        ("vit_b_16", "baseline"),
    ]
    
    for model_name, variant in models_to_test:
        print(f"\n=== Testing {model_name} with {variant} variant ===")
        try:
            model = create_vit_with_alc(
                model_name=model_name, 
                variant=variant,
                num_classes=1000,
                extract_layer=-2 if variant == "multiscale" else None,
                upsample_factor=2 if variant == "highres" else None
            )
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            print(f"Output shape: {output.shape}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nAll configurations tested successfully!")