import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50
from torchvision.ops import DeformConv2d
from AdaptiveLocal2DLayerv2 import AdaptiveLocal2DLayer, AdaptiveLocal2DLayer_SeparateChannels


def make_alc2d(input_size, output_size, channel_separate=False, si_init=(0.03, 0.03), shared_weights=False):
    if channel_separate:
       return AdaptiveLocal2DLayer_SeparateChannels(input_size, output_size,
                                               normed=True, 
                                               si_init=si_init,
                                               shared_weights=shared_weights, activ='relu')
    
    else:
            
        return AdaptiveLocal2DLayer(input_size, output_size,
                                           normed=True, layer_norm=False,
                                           n_embedding=None,
                                           si_init=si_init,
                                           shared_weights=shared_weights, activ='relu')


def make_resnet_layer3_features(base):
    return nn.Sequential(
        base.conv1,
        base.bn1,
        base.relu,
        base.maxpool,
        base.layer1,
        base.layer2,
        base.layer3,
    )


def infer_feature_shape(features):
    dummy_input = torch.zeros((1, 3, 224, 224))
    with torch.no_grad():
        feature_map = features(dummy_input)
    _, channels, height, width = feature_map.shape
    print(f"Feature map shape: {channels}x{height}x{width}")
    return channels, height, width


class LocalConnected1x1(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, bias=True):
        super().__init__()
        self.height = height
        self.width = width
        self.weight = nn.Parameter(torch.empty(height, width, out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(height, width, out_channels))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        output = torch.einsum('bchw,hwoc->bohw', x, self.weight)
        if self.bias is not None:
            output = output + self.bias.permute(2, 0, 1).unsqueeze(0)
        return output


class CoordConvHead2D(nn.Module):
    def __init__(self, in_channels, height, width, hidden_channels=256):
        super().__init__()
        y_coords = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1).expand(1, 1, height, width)
        x_coords = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('coord_grid', torch.cat([x_coords, y_coords], dim=1))
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        coords = self.coord_grid.expand(x.size(0), -1, -1, -1)
        return self.proj(torch.cat([x, coords], dim=1))


class DeformConvHead2D(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        offset_channels = 2 * kernel_size * kernel_size
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.offset = nn.Conv2d(hidden_channels, offset_channels, kernel_size=3, padding=1)
        self.deform = DeformConv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.reduce(x)
        offsets = self.offset(x)
        x = self.deform(x, offsets)
        x = self.norm(x)
        return self.act(x)


class ResNetWithCoordConvHead14x14(nn.Module):
    def __init__(self, base, num_classes, hidden_channels=256, head_dropout=0.0):
        super().__init__()
        self.features = make_resnet_layer3_features(base)
        channels, height, width = infer_feature_shape(self.features)
        self.head = CoordConvHead2D(channels, height, width, hidden_channels=hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class ResNetWithLocalConnectedHead14x14(nn.Module):
    def __init__(self, base, num_classes, bottleneck_channels=128, head_dropout=0.0):
        super().__init__()
        self.features = make_resnet_layer3_features(base)
        channels, height, width = infer_feature_shape(self.features)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        self.local = nn.Sequential(
            LocalConnected1x1(bottleneck_channels, bottleneck_channels, height, width),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.fc = nn.Linear(bottleneck_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        x = self.local(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class ResNetWithDeformConvHead14x14(nn.Module):
    def __init__(self, base, num_classes, hidden_channels=256, head_dropout=0.0):
        super().__init__()
        self.features = make_resnet_layer3_features(base)
        channels, _, _ = infer_feature_shape(self.features)
        self.head = DeformConvHead2D(channels, hidden_channels=hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class ResNetWithALC_14x14(nn.Module):
    """Extract features before the last residual block to get 14x14 feature maps"""
    def __init__(self, base, num_classes, alc_output=(32, 32), si_init=(0.03, 0.03), head_dropout=0.0):
        super().__init__()
        self.features = make_resnet_layer3_features(base)
        C, H, W = infer_feature_shape(self.features)
        
        self.alc = make_alc2d((C, H, W), output_size=alc_output, si_init=si_init)
        self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.fc = nn.Linear(np.prod(alc_output), num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.alc(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
    
    
class ResNetWithALC_14x14_ChannelSep(nn.Module):
    """Extract features before the last residual block to get 14x14 feature maps"""
    def __init__(self, base, num_classes, spatial_output=(7, 7), si_init=(0.03, 0.03), head_dropout=0.0):
        super().__init__()
        self.features = make_resnet_layer3_features(base)
        C, H, W = infer_feature_shape(self.features)
        
        alc_output = (C, spatial_output[0], spatial_output[1])
        self.alc = make_alc2d((C, H, W), output_size=alc_output, channel_separate=True, si_init=si_init)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # or use flattening for comparison
        self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.alc(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

class ResNetWithALC_28x28(nn.Module):
    """Extract features before layer3 to get 28x28 feature maps"""
    def __init__(self, base, num_classes):
        super().__init__()
        # Stop after layer2 to get 512x28x28
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,  # 256x56x56
            base.layer2,  # 512x28x28
            # Skip layer3 and layer4
        )
        
        # Calculate feature map dimensions
        input_size = (1, 3, 224, 224)
        dummy_input = torch.zeros(input_size)
        with torch.no_grad():
            x = self.features(dummy_input)
        B, C, H, W = x.shape
        print(f"Feature map shape: {C}x{H}x{W}")
        
        alc_output = (32, 32)
        self.alc = make_alc2d((C, H, W), output_size=alc_output)
        self.fc = nn.Linear(np.prod(alc_output), num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.alc(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class ResNetWithALC_ModifiedStride(nn.Module):
    """Modify stride in layer4 to keep 14x14 resolution"""
    def __init__(self, base, num_classes):
        super().__init__()
        
        # Create a modified version of the base model
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        
        # Modify layer4 to use stride=1 instead of stride=2
        self.layer4 = self._modify_layer4_stride(base.layer4)
        
        # Calculate feature map dimensions
        input_size = (1, 3, 224, 224)
        dummy_input = torch.zeros(input_size)
        with torch.no_grad():
            x = self._extract_features(dummy_input)
        B, C, H, W = x.shape
        print(f"Feature map shape: {C}x{H}x{W}")
        
        alc_output = (32, 32)
        self.alc = make_alc2d((C, H, W), output_size=alc_output)
        self.fc = nn.Linear(np.prod(alc_output), num_classes)
    
    def _modify_layer4_stride(self, layer4):
        """Modify the first block of layer4 to use stride=1"""
        modified_layer4 = nn.Sequential()
        for i, block in enumerate(layer4):
            if i == 0:  # First block
                # Create a new block with stride=1
                new_block = type(block)(
                    block.conv1.in_channels,
                    block.conv1.out_channels,
                    stride=1,  # Change from 2 to 1
                    downsample=None if block.downsample is None else self._modify_downsample(block.downsample)
                )
                # Copy weights
                new_block.load_state_dict(block.state_dict(), strict=False)
                modified_layer4.add_module(str(i), new_block)
            else:
                modified_layer4.add_module(str(i), block)
        return modified_layer4
    
    def _modify_downsample(self, downsample):
        """Modify downsample to use stride=1"""
        if downsample is None:
            return None
        conv, bn = downsample
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, 1, stride=1, bias=False)
        new_conv.weight.data = conv.weight.data
        return nn.Sequential(new_conv, bn)
    
    def _extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward(self, x):
        x = self._extract_features(x)
        x = self.alc(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class ResNetWithALC_FeaturePyramid(nn.Module):
    """Use multiple feature maps from different layers"""
    def __init__(self, base, num_classes):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool, base.layer1
        )  # 256x56x56
        self.layer2 = base.layer2  # 512x28x28
        self.layer3 = base.layer3  # 1024x14x14
        
        # You can choose which feature map to use
        self.use_layer = 2  # 1 for 56x56, 2 for 28x28, 3 for 14x14
        
        # Calculate dimensions based on chosen layer
        if self.use_layer == 1:
            feature_channels, H, W = 256, 56, 56
        elif self.use_layer == 2:
            feature_channels, H, W = 512, 28, 28
        else:  # layer 3
            feature_channels, H, W = 1024, 14, 14
            
        print(f"Using layer {self.use_layer}: {feature_channels}x{H}x{W}")
        
        alc_output = (32, 32)
        self.alc = make_alc2d((feature_channels, H, W), output_size=alc_output)
        self.fc = nn.Linear(np.prod(alc_output), num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        if self.use_layer == 1:
            features = x
        else:
            x = self.layer2(x)
            if self.use_layer == 2:
                features = x
            else:
                x = self.layer3(x)
                features = x
        
        x = self.alc(features)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNetChopHead14x14(nn.Module):
    """ResNet with layer4 removed, no ALC, standard global pooling + linear head"""
    def __init__(self, base, num_classes, head_dropout=0.0):
        super().__init__()
        
        # Keep layers up to layer3 (same as ResNetWithALC_14x14)
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3
        )
        
        # Feature shape: (B, 1024, 14, 14)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # or use flattening for comparison
        self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.features(x)             # (B, 1024, 14, 14)
        x = self.pool(x)                 # (B, 1024, 1, 1)
        x = torch.flatten(x, 1)          # (B, 1024)
        x = self.dropout(x)
        return self.fc(x)


# Example usage:
def create_model_with_larger_features(model_type="14x14"):
    """
    Create ResNet model with larger feature maps
    
    Args:
        model_type: "14x14", "28x28", "modified_stride", or "pyramid"
    """
    base = resnet50(weights="DEFAULT")
    num_classes = 1000
    
    if model_type == "14x14":
        from torchinfo import summary
        model = ResNetWithALC_14x14(base, num_classes)
        summary(model,input_size=(1,3,224,224),  col_names=["input_size","kernel_size", "output_size", "num_params"])
        model = ResNetWithALC_14x14_ChannelSep(base, num_classes)
        summary(model,input_size=(1,3,224,224),  col_names=["input_size","kernel_size", "output_size", "num_params"])
        input("Herere")
    elif model_type == "28x28":
        model = ResNetWithALC_28x28(base, num_classes)
    elif model_type == "modified_stride":
        model = ResNetWithALC_ModifiedStride(base, num_classes)
    elif model_type == "pyramid":
        model = ResNetWithALC_FeaturePyramid(base, num_classes)
    else:
        raise ValueError("Invalid model_type")
    
    return model

# Test the different approaches
if __name__ == "__main__":
    # Assuming make_alc2d is defined elsewhere
    
    
    print("Testing different feature map sizes:")
    
    print("\n1. 14x14 feature maps (1024 channels):")
    model_14x14 = create_model_with_larger_features("14x14")
    
    print("\n2. 28x28 feature maps (512 channels):")
    model_28x28 = create_model_with_larger_features("28x28")
    
    print("\n3. Modified stride approach:")
    # model_modified = create_model_with_larger_features("modified_stride")
    
    print("\n4. Feature pyramid approach:")
    model_pyramid = create_model_with_larger_features("pyramid")