from easydict import EasyDict

cfg = EasyDict()

# Record
cfg.RECORD = EasyDict()
cfg.RECORD.LOG = "./experiments"

# Dataset
cfg.DATA = EasyDict()
cfg.DATA.ROOT = "./AuViDataset"
cfg.DATA.NUM_FRAMES = 16
cfg.DATA.USE_SOUND = True
cfg.DATA.RESOLUTION = (224,384)

# Train
cfg.TRAIN = EasyDict()
cfg.TRAIN.BATCH_SIZE = 2

# Solver
cfg.SOLVER = EasyDict()
cfg.SOLVER.LR = 1e-4
cfg.SOLVER.MIN_LR = 1e-5
cfg.SOLVER.MAX_EPOCH = 120
cfg.SOLVER.OPTIMIZING_METHOD = 'adamw'
cfg.SOLVER.MONITORED_EPOCHES = [ i for i in range(60,121,20)]

# Model
_MOTION_ENCODERS = ('mvitv2s','s3d', 'slowfast4x16', 'morphmlps', 'uniformerb', 'videoswins', 'x3dl')
_MOTION_WEIGHTS = {
    'mvitv2s': "./weights/MViTv2_S_16x4_k400_f302660347.pyth",
    's3d': "./weights/S3D_kinetics400_rm_fc.pt",
    'slowfast4x16': "./weights/SLOWFAST_4x16_R50.pkl",
    'morphmlps': "./weights/mlp_s16x4_k400.pth",
    'uniformerb': "./weights/uniformer_base_k400_16x4.pth",
    'videoswins': "./weights/swin_small_patch244_window877_kinetics400_1k.pth",
    'x3dl': "./weights/x3d_l.pyth",
}
_LATERAL_BOOL = {
    'mvitv2s': [True, True, True, True],
    's3d': [True, True, False, False],
    'slowfast4x16': [False, False, False, False],
    'morphmlps': [True, True, True, True],
    'uniformerb': [True, True, True, True],
    'videoswins': [True, True, True, True],
    'x3dl': [True, True, True, True],
}
_NUM_VIS_TOKENS = {
    'mvitv2s': 8 * 7 * 12,
    's3d': 4 * 7 * 7,
    'slowfast4x16': 4 * 7 * 7,
    'morphmlps': 8 * 7 * 7,
    'uniformerb': 8 * 7 * 7,
    'videoswins': 8 * 7 * 7,
    'x3dl': 16 * 7 * 7,
}

# select your model
_model_name = _MOTION_ENCODERS[0]

cfg.MODEL = EasyDict()
cfg.MODEL.LATERAL_BOOL = _LATERAL_BOOL[_model_name]
cfg.MODEL.LATERAL_STRIDE = [4, 4, 4, 4] if _model_name == 'x3dl' else [2, 2, 2, 2]
cfg.MODEL.MOTION_ENCODER = _model_name
cfg.MODEL.MOTION_ENCODER_WEIGHT = _MOTION_WEIGHTS[_model_name]
cfg.MODEL.MOTION_ENCODER_EMBEDS = {
    'mvitv2s': (96, 192, 384, 768),
    's3d': (192, 480, 832, 1024),
    'slowfast4x16': (320, 640, 1280, 2048),
    'morphmlps': (112, 224, 392, 784),
    'uniformerb': (64, 128, 320, 512),
    'videoswins': (96, 192, 384, 768),
    'x3dl': (24, 48, 96, 192),
}
cfg.MODEL.NUM_VIS_TOKENS = _NUM_VIS_TOKENS
cfg.MODEL.IMAGE_SALIENCY_ENCODER_WEIGHT = "./weights/image_saliency_encoder_convnext_tiny.pt"
cfg.MODEL.AUDIO_ENCODER_WEIGHT = "./weights/resnet18_vggsound.pt"

#S3D
cfg.MODEL.S3D = EasyDict()
cfg.MODEL.S3D.POOL_STRIDE = 1

#MViTv2
cfg.MODEL.MVIT2 = EasyDict()
cfg.MODEL.MVIT2.PATH_CFG = ["./configs/MVITv2_S_16x4.yaml"]

#Slowfast
cfg.MODEL.SLOWFAST = EasyDict()
cfg.MODEL.SLOWFAST.PATH_CFG = ["./configs/SLOWFAST_4x16_R50.yaml"]

#MorphMLP
cfg.MODEL.MORPH = EasyDict()
cfg.MODEL.MORPH.PATH_CFG = "./configs/K400_MLP_S16x4.yaml"

#X3D
cfg.MODEL.X3D = EasyDict()
cfg.MODEL.X3D.PATH_CFG = ["./configs/X3D_L.yaml"]

#Uniformer
cfg.MODEL.UNIFORMER = EasyDict()
cfg.MODEL.UNIFORMER.PATH_CFG = "./configs/uniformer_b16x4_k400.yaml"




