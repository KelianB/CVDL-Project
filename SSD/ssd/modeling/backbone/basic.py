import torch
from torch import nn

from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

"""
ResNet architectures (from torchvision):

resnet18: BasicBlock, [2, 2, 2, 2]
resnet34: BasicBlock, [3, 4, 6, 3]
resnet50: Bottleneck, [3, 4, 6, 3]
resnet101: Bottleneck, [3, 4, 23, 3]
resnet152: Bottleneck, [3, 8, 36, 3]
resnext50_32x4d: Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4
resnext101_32x8d: Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8
wide_resnet50_2: Bottleneck, [3, 4, 6, 3], width_per_group=64*2
wide_resnet101_2: Bottleneck, [3, 4, 23, 3], width_per_group=64*2
"""


""" Conv2d - BatchNorm2d - ReLU block """
def conv_bn_relu(channels_in, channels_out, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=1, padding=padding),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True)
    )


"""
    Utility model used to indicate where to output feature maps.
    Optionally, if given a module, it will first apply it to the input.
"""
class OutputMarker(nn.Module):
    def __init__(self, before_output=None):
        super(OutputMarker, self).__init__()
        self.before_output = before_output
    
    def forward(self, x):
        return x if self.before_output == None else self.before_output(x)

 
"""
    A resnet34-based backbone for SSD.
    Uses the same layer setups as the implementation seen here:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    This feature extractor outputs a list of 6 feature maps. See config for sizes.
"""    
class BasicModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        # Bottleneck or BasicBlock
        block = BasicBlock
        self.inplanes = 64
        
        # The architecture from which to load weights
        pretrained_arch = "resnet34" if cfg.MODEL.BACKBONE.PRETRAINED else None
        
        def resnet_layer(blocks, planes, stride):
            return self._make_layer(block, planes, blocks, stride=stride)

        # Pre-define modules for which we have pre-trained weights
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = resnet_layer(3, 64, 1)  
        self.layer2 = resnet_layer(4, 128, 2)
        self.layer3 = resnet_layer(6, 256, 2)
        self.layer4 = resnet_layer(3, 512, 2)
        
        # Define architecture
        self.architecture = nn.ModuleList([
            nn.Sequential(
                self.conv1,
                self.bn1,
                nn.ReLU(inplace=True),
                #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ),
            self.layer1,
            self.layer2,
            self.layer3,
            OutputMarker(conv_bn_relu(256, 128)),
            self.layer4,

            resnet_layer(2, 256, 1), 
            OutputMarker(conv_bn_relu(256, 256)),
            
            resnet_layer(3, 1024, 2),
            OutputMarker(conv_bn_relu(1024, 512)),
        
            resnet_layer(4, 512, 2),
            OutputMarker(conv_bn_relu(512, 256)),
            
            resnet_layer(4, 128, 2),
            OutputMarker(),
            
            resnet_layer(4, 64, 1),
            conv_bn_relu(64, 64, kernel_size=3, padding=0),
            OutputMarker()
        ])

        # Inititialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
        # Load pre-trained weights if required
        if pretrained_arch != None:
            state_dict = load_state_dict_from_url(model_urls[pretrained_arch])
            self.load_state_dict(state_dict, strict=False)

            
    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

        
    def forward(self, x):
        out_features = []
        for l in list(self.architecture):
            if isinstance(l, OutputMarker):
                out_features.append(l(x))
            else:
                x = l(x)
    
        # Check feature map dimensions
        for idx, feature in enumerate(out_features):
            out_channel = self.output_channels[idx]
            feature_map_size = self.output_feature_size[idx] 
            expected_shape = (out_channel, feature_map_size[1], feature_map_size[0])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        
        return out_features