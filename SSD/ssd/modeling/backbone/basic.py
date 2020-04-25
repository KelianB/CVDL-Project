import torch
from torch import nn
from ssd import torch_utils

from ssd.modeling.backbone.resnet import ResNetFeatureExtractor, BasicBlock, Bottleneck, OutputMarker

class BasicModel(nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        block = BasicBlock
        self.feature_extractor = ResNetFeatureExtractor(cfg)
        #print(self.feature_extractor)
        
    def forward(self, x):
        out_features = []
        
        for i, l in enumerate(list(self.feature_extractor.architecture)):
            #print("\nlayer", i)
            if isinstance(l, OutputMarker):
                out_features.append(l(x))
            else:
                x = l(x)
            #print("\nlayer", i, "done")
            
        # Check feature map dimensions
        for idx, feature in enumerate(out_features):
            out_channel = self.output_channels[idx]
            feature_map_size = self.output_feature_size[idx] 
            expected_shape = (out_channel, feature_map_size[1], feature_map_size[0])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        
        return out_features