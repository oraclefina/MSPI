from backbones.s3d import S3D_features_only
from backbones.MViT import MViT
from backbones.sf import SlowFast
from backbones.X3D import X3D
from backbones.video_swin_transformer import SwinTransformer3D
from backbones.uniformer import Uniformer
from backbones.MorphMLP import MorphMLP_32_features_only

_MOTION_ENCODERS = ('mvitv2s','s3d', 'slowfast4x16', 'morphmlps', 'uniformerb', 'videoswins', 'x3dl')

def video_motion_extractor(cfg):
    motion_encoder = None
    if cfg.MODEL.MOTION_ENCODER == 's3d':
        motion_encoder = S3D_features_only(pool=cfg.MODEL.S3D.POOL_STRIDE)
    elif cfg.MODEL.MOTION_ENCODER == 'mvitv2s':
        motion_encoder = MViT(path_to_configs=cfg.MODEL.MVIT2.PATH_CFG)
    elif cfg.MODEL.MOTION_ENCODER == 'slowfast4x16':
        motion_encoder = SlowFast(path_to_config=cfg.MODEL.SLOWFAST.PATH_CFG)
    elif cfg.MODEL.MOTION_ENCODER == 'morphmlps':
        motion_encoder = MorphMLP_32_features_only(path_to_config=cfg.MODEL.MORPH.PATH_CFG)
    elif cfg.MODEL.MOTION_ENCODER == 'uniformerb':
        motion_encoder = Uniformer(yaml_path=cfg.MODEL.UNIFORMER.PATH_CFG)
    elif cfg.MODEL.MOTION_ENCODER == 'videoswins':
        motion_encoder = SwinTransformer3D()
    elif cfg.MODEL.MOTION_ENCODER == 'x3dl':
        motion_encoder = X3D(path_to_config=cfg.MODEL.X3D.PATH_CFG)

    if motion_encoder is None:
        raise Exception("Invalid Motion Encoder!")

    return motion_encoder