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
        image_size = cfg.INPUT.IMAGE_SIZE
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        #block = Bottleneck
        block = BasicBlock
        
        #blocks_per_layers = [3, 4, 6, 3, 3, 3, 3]
        #blocks_per_layers = [6, 4, 6, 8, 10, 10, 6]
        #blocks_per_layers = [3, 4, 6, 10, 10, 10, 10]
        #blocks_per_layers = [4, 8, 10, 6, 6, 4, 3]
        blocks_per_layers = [3, 4, 6, 2, 2, 2, 2, 2]
        self.feature_extractor = ResNetFeatureExtractor(block, blocks_per_layers, pretrained_arch="resnet34")
        
        #print(self.feature_extractor)
        
        self._init_weights()

    def _init_weights(self):
        """
        layers = [*self.additional_blocks]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)
        """
        
    def forward(self, x):
        out_features = []
        
        for i, l in enumerate(list(self.feature_extractor.architecture)):
            #print("\nlayer", i)
            x = l(x)
            #print("\nlayer", i, "done")
            if isinstance(l, OutputMarker):
                out_features.append(x)
           
        # Check feature map dimensions
        for idx, feature in enumerate(out_features):
            out_channel = self.output_channels[idx]
            feature_map_size = self.output_feature_size[idx] 
            expected_shape = (out_channel, feature_map_size[1], feature_map_size[0])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        
        return out_features
