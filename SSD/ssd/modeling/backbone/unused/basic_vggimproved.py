import torch
from torch import nn
from ssd import torch_utils

import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import os
import itertools
from collections import OrderedDict

class VGG16(nn.Module):
    '''
    input image: BGR format, range [0, 255], then subtract mean
    '''
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv1_1 = nn.Conv2d(  3,  64, 3, padding=1)
        self.conv1_2 = nn.Conv2d( 64,  64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)

        self.conv2_1 = nn.Conv2d( 64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, ceil_mode=True)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(512, 1024, 3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, 1)

    def load_pretrained(self, path):
        weights = torch.load(path)

        lookup = {'conv1_1':'0', 'conv1_2':'2', 'conv2_1':'5', 'conv2_2':'7', 
                  'conv3_1':'10', 'conv3_2':'12', 'conv3_3':'14', 
                  'conv4_1':'17', 'conv4_2':'19', 'conv4_3':'21',
                  'conv5_1':'24', 'conv5_2':'26', 'conv5_3':'28',
                  'conv6':'31', 'conv7':'33'}

        model_dict = self.state_dict()
        pretrained_dict = {}
        for name, ind in lookup.items():
            for ext in ['.weight', '.bias']:
                pretrained_dict[name + ext] = weights[ind + ext]
        model_dict.update(pretrained_dict)

        self.load_state_dict(model_dict)

        
        
        
class L2Norm(nn.Module):
    def __init__(self,n_channels, scale=20):
        super(L2Norm,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        nn.init.constant(self.weight, scale)

    def forward(self, x):
        x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


    
    
    
class BasicModel(nn.Module):

    config = {
        'name': 'SSD300-VGG16',
        'aspect_ratios': ((1/2.,  1,  2), 
                          (1/3.,  1/2.,  1,  2,  3), 
                          (1/3.,  1/2.,  1,  2,  3), 
                          (1/3.,  1/2.,  1,  2,  3), 
                          (1/2.,  1,  2),
                          (1/2.,  1,  2)),
    }
    
    
    def __init__(self, cfg):
        super().__init__()
        self.n_classes = 4

        self.Base = VGG16()
        self.Extra = nn.Sequential(OrderedDict([
            ('extra1_1', nn.Conv2d(1024, 256, 1)),
            ('extra1_2', nn.Conv2d(256, 512, 3, padding=1, stride=2)),
            ('extra2_1', nn.Conv2d(512, 128, 1)),
            ('extra2_2', nn.Conv2d(128, 256, 3, padding=1, stride=2)),
            ('extra3_1', nn.Conv2d(256, 128, 1)),
            ('extra3_2', nn.Conv2d(128, 256, 3)),
            ('extra4_1', nn.Conv2d(256, 128, 1)),
            ('extra4_2', nn.Conv2d(128, 256, 3)),
        ]))
        self.pred_layers = ['conv4_3', 'conv7', 'extra1_2', 'extra2_2', 'extra3_2', 'extra4_2']
        n_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        
        self.L2Norm = nn.ModuleList([L2Norm(512, 20)])
        self.norm_layers = ['conv4_3']   # decrease prediction layers' influence on backbone

        self.Loc = nn.ModuleList([])
        self.Conf = nn.ModuleList([])
        for i, ar in enumerate(self.config['aspect_ratios']):
            n = len(ar) + 1
            self.Loc.append(nn.Conv2d(n_channels[i], n * 4, 3, padding=1))
            self.Conf.append(nn.Conv2d(n_channels[i], n * (self.n_classes + 1), 3, padding=1))
            
        self.relu = nn.ReLU()

    def forward(self, x):
        xs = []
        for name, m in itertools.chain(self.Base._modules.items(), 
                                       self.Extra._modules.items()):
            if isinstance(m, nn.Conv2d):
                x = F.relu(m(x), inplace=True)
            else:
                x = m(x)
            
            if name in self.pred_layers:
                if name in self.norm_layers:
                    i = self.norm_layers.index(name)
                    xs.append(self.L2Norm[i](x))
                else:
                    xs.append(x)

                
        return xs
        #return self._prediction(xs)

    def _prediction(self, xs):
        locs = []
        confs = []
        for i, x in enumerate(xs):
            loc = self.Loc[i](x)
            loc = loc.permute(0, 2, 3, 1).contiguous().view(loc.size(0), -1, 4)
            locs.append(loc)

            conf = self.Conf[i](x)
            conf = conf.permute(0, 2, 3, 1).contiguous().view(conf.size(0), -1, self.n_classes + 1)
            confs.append(conf)
        return torch.cat(locs, dim=1), torch.cat(confs, dim=1)


    def init_parameters(self, backbone=None):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                m.bias.data.zero_()
        self.apply(weights_init)

        if backbone is not None and os.path.isfile(backbone):
            self.Base.load_pretrained(backbone)
            print('Backbone loaded!')
        else:
            print('No backbone file!')
    
    
    


class BasicModelOld(nn.Module):
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
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        # Custom backbone
        feature_bank_extractors = nn.Sequential(
            nn.Sequential(
                # Pad 320x240 to square (320x320)
                # nn.Conv2d(in_channels=image_channels, out_channels=image_channels, kernel_size=1, stride=1, padding=(40,0)),# 
                
                nn.Conv2d(in_channels=image_channels, out_channels=16, kernel_size=4, stride=1, padding=2),
                nn.ReLU(),

                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=output_channels[0], kernel_size=3, stride=2, padding=1),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(in_channels=output_channels[0], out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=128, out_channels=output_channels[1], kernel_size=3, stride=2, padding=1),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=output_channels[1], out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=256, out_channels=output_channels[2], kernel_size=3, stride=2, padding=1),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=output_channels[2], out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=256, out_channels=output_channels[3], kernel_size=3, stride=2, padding=1),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=output_channels[3], out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Conv2d(in_channels=128, out_channels=output_channels[4], kernel_size=3, stride=2, padding=1),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=output_channels[4], out_channels=128, kernel_size=(2,3), stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                nn.Dropout2d(p=0.2),
                nn.Conv2d(in_channels=128, out_channels=output_channels[5], kernel_size=3, stride=2, padding=0),
            )
        )
        
        
        self.feature_bank_extractors = torch_utils.to_cuda(feature_bank_extractors)

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        
        out_features = []
        for seq in self.feature_bank_extractors:
            x = seq(x)
            out_features.append(x)
        
        for idx, feature in enumerate(out_features):
            out_channel = self.output_channels[idx]
            feature_map_size = self.output_feature_size[idx] 
            expected_shape = (out_channel, feature_map_size[1], feature_map_size[0])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

