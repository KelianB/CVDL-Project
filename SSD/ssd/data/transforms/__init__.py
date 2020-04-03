from ssd.modeling.box_head.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            ToPercentCoords(),
            NonSquareResize(cfg.INPUT.IMAGE_SIZE[0], cfg.INPUT.IMAGE_SIZE[1]),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            NonSquareResize(cfg.INPUT.IMAGE_SIZE[0], cfg.INPUT.IMAGE_SIZE[1]),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
