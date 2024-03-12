from types import SimpleNamespace
import torch

Config = SimpleNamespace(
    LOG_DIR = 'training_results',
    TAG = 'fullsegformer_tiny_v2',
    WANDB_PROJECT = 'new_code_tests',

    ACCELERATOR = "cuda",
    GPUS = 0,  # which gpus to use
    DEVICES = "auto", # how many gpus to use, auto for all available
    PRECISION = '16-mixed',  # 16-mixed or 32bit
    BATCHSIZE = 1,
    EPOCHS = 20,
    
    OPTIMIZER = SimpleNamespace(
        TYPE = 'AdamW',
        # LR = 3e-4,
        LR = 6e-5,
        # WEIGHT_DECAY = 1e-7,
        WEIGHT_DECAY = 2e-3,
    ),

    # GRAD_NORM_CLIP = 5,
    GRAD_NORM_CLIP = 0.5,

    N_WORKERS = 1,
    VIS_INTERVAL = 500,
    LOGGING_INTERVAL = 100,

    PRETRAINED = SimpleNamespace(
        LOAD_WEIGHTS = False,
        RESUME_TRAINING = False,
        PATH = '/home/perception/workspace/training_results/<>/',
        CKPT = 'checkpoints/model-epoch=<>.ckpt',
    ),
    
    DATASET = SimpleNamespace(
        DATAROOT = '/home/perception/Datasets/nuscenes/mini',
        # VERSION = 'trainval',
        VERSION = 'v1.0-mini',
        NAME = 'nuscenes',
        IGNORE_INDEX = 255,  # Ignore index when creating flow/offset labels
        FILTER_INVISIBLE_VEHICLES = True,  # Filter vehicles that are not visible from the cameras
        N_CAMERAS = 6,  # Number of cameras
    ),

    TIME_RECEPTIVE_FIELD = 3,  # how many frames of temporal context (1 for single timeframe)
    N_FUTURE_FRAMES = 4,  # how many time steps into the future to predict

    IMAGE = SimpleNamespace(
        FINAL_DIM = (224, 480),
        ORIGINAL_DIM = (900, 1600),
        RESIZE_SCALE = 0.3,
        TOP_CROP = 46,
        ORIGINAL_HEIGHT = 900 ,  # Original input RGB camera height
        ORIGINAL_WIDTH = 1600 ,  # Original input RGB camera width
        NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    ),

    LIFT = SimpleNamespace(
        # Long BEV dimensions
        X_BOUND = [-50.0, 50.0, 0.5],  # Forward
        Y_BOUND = [-50.0, 50.0, 0.5],  # Sides
        Z_BOUND = [-10.0, 10.0, 20.0],  # Height
        D_BOUND = [2.0, 50.0, 1.0],

        # Short BEV dimensions
        # X_BOUND: [-15.0, 15.0, 0.15]  # Forward
        # Y_BOUND: [-15.0, 15.0, 0.15]  # Sides
        # Z_BOUND = [-10.0, 10.0, 20.0],  # Height
        # D_BOUND = [2.0, 50.0, 1.0],
    ),

    MODEL = SimpleNamespace(
        NAME = 'full_segformer',  # 'powerbev', 'powerformer', 'powerformer_dualenc' or 'full_segformer'
        ENCODER = SimpleNamespace(
            DOWNSAMPLE = 8,
            NAME = 'efficientnet-b4',
            OUT_CHANNELS = 64,
            USE_DEPTH_DISTRIBUTION = True,
        ),
        STCONV = SimpleNamespace(
            LATENT_DIM = 16, # 16
            NUM_FEATURES = [16, 24, 32, 48, 64],
            NUM_BLOCKS = 4,
            INPUT_EGOPOSE = True,
        ),
        # Tiny
        # SEGFORMER = SimpleNamespace(
        #     N_ENCODER_BLOCKS = 5,
        #     DEPTHS = [2, 2, 2, 2, 2],
        #     SEQUENCE_REDUCTION_RATIOS = [8, 4, 2, 1, 1],
        #     HIDDEN_SIZES = [16, 24, 32, 48, 64],  # Must be equal to STCONV.NUMFEATURES
        #     PATCH_SIZES = [7, 3, 3, 3, 3],
        #     STRIDES = [2, 2, 2, 2, 2],
        #     NUM_ATTENTION_HEADS = [1, 2, 4, 8, 8],
        #     MLP_RATIOS = [4, 4, 4, 4, 4],
        #     HEAD_DIM_MULTIPLIER = 4,
        #     HEAD_KERNEL = 2,
        #     HEAD_STRIDE = 2,
        # ),
   
        # B0
        SEGFORMER = SimpleNamespace(
            N_ENCODER_BLOCKS = 5,
            DEPTHS = [2, 2, 2, 2, 2],
            SEQUENCE_REDUCTION_RATIOS = [8, 8, 4, 2, 1],
            HIDDEN_SIZES = [16, 32, 64, 160, 256],  # Must be equal to STCONV.NUMFEATURES
            PATCH_SIZES = [7, 7, 3, 3, 3],
            STRIDES = [2, 2, 2, 2, 2],
            NUM_ATTENTION_HEADS = [1, 1, 2, 4, 8],
            MLP_RATIOS = [4, 4, 4, 4, 4],
            HEAD_DIM_MULTIPLIER = 4,
            HEAD_KERNEL = 2,
            HEAD_STRIDE = 2,
        ),
        # SEGFORMER = SimpleNamespace(
        #     N_ENCODER_BLOCKS = 4,
        #     DEPTHS = [2, 2, 2, 2],
        #     SEQUENCE_REDUCTION_RATIOS = [8, 4, 2, 1],
        #     HIDDEN_SIZES = [32, 64, 160, 256],  # Must be equal to STCONV.NUMFEATURES
        #     PATCH_SIZES = [7, 3, 3, 3],
        #     STRIDES = [4, 2, 2, 2],
        #     NUM_ATTENTION_HEADS = [1, 2, 4, 8],
        #     MLP_RATIOS = [4, 4, 4, 4],
        #     HEAD_DIM_MULTIPLIER = 4,
        #     HEAD_KERNEL = 4,
        #     HEAD_STRIDE = 4,
        # ),

        # B1
        # SEGFORMER = SimpleNamespace(
        #     N_ENCODER_BLOCKS = 5,
        #     DEPTHS = [2, 2, 2, 2, 2],
        #     SEQUENCE_REDUCTION_RATIOS = [8, 8, 4, 2, 1],
        #     HIDDEN_SIZES = [64, 64, 128, 320, 512],  # Must be equal to STCONV.NUMFEATURES
        #     PATCH_SIZES = [7, 7, 3, 3, 3],
        #     STRIDES = [2, 2, 2, 2, 2],
        #     NUM_ATTENTION_HEADS = [1, 1, 2, 4, 8],
        #     MLP_RATIOS = [4, 4, 4, 4, 4],
        #     HEAD_DIM_MULTIPLIER = 4,
        #     HEAD_KERNEL = 2,
        #     HEAD_STRIDE = 2,
        # ),        
        
        TEMPORAL_MODEL = SimpleNamespace(
            NAME = 'temporal_block',  # type of temporal model
            START_OUT_CHANNELS = 64,
            EXTRA_IN_CHANNELS = 0,
            INBETWEEN_LAYERS = 0,
            PYRAMID_POOLING = True,
            INPUT_EGOPOSE = True,
        ),
        DISTRIBUTION = SimpleNamespace(
            LATENT_DIM = 32,
            MIN_LOG_SIGMA = -5.0,
            MAX_LOG_SIGMA = 5.0,
        ),
        FUTURE_PRED = SimpleNamespace(
            N_GRU_BLOCKS = 3,
            N_RES_LAYERS = 3,
        ),

        BN_MOMENTUM = 0.1,
        SUBSAMPLE = False,  # Subsample frames for Lyft
    ),

    SEMANTIC_SEG = SimpleNamespace(
        WEIGHTS = [1.0, 2.0],  # per class cross entropy weights (bg, dynamic, drivable, lane)
        USE_TOP_K = True,  # backprop only top-k hardest pixels
        TOP_K_RATIO = 0.25,
    ),

    INSTANCE_SEG = SimpleNamespace(),

    INSTANCE_FLOW = SimpleNamespace(
        ENABLED = True,
    ),

    PROBABILISTIC = SimpleNamespace(
        ENABLED = False,  # learn a distribution over futures
        WEIGHT = 100.0,
        FUTURE_DIM = 6,  # number of dimension added (future flow, future centerness, offset, seg)
    ),

    FUTURE_DISCOUNT = 0.95,

    VISUALIZATION = SimpleNamespace(
        OUTPUT_PATH = './visualization_outputs',
        SAMPLE_NUMBER = 1000,
        VIS_GT = True,
    )
)
