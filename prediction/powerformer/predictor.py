import torch
import torch.nn as nn
import numpy as np
import time

# from transformers import TimesformerConfig, TimesformerModel
# from transformers import PvtModel, PvtConfig
from transformers import SegformerModel, SegformerConfig, SegformerForSemanticSegmentation

from prediction.powerformer.feature_extractor import FeatureExtractor
from prediction.powerformer.stconv import MultiBranchSTconv, MultiBranchSTconv_dualencoder


# from custom_pred.model.transformer_modules import Longformer

class PowerFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.feature_width = int((self.cfg.LIFT.X_BOUND[1] - self.cfg.LIFT.X_BOUND[0])\
                              /self.cfg.LIFT.X_BOUND[2])

        self.use_ego_motion = self.cfg.MODEL.STCONV.INPUT_EGOPOSE
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD

        self.temporal_attn_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS
        if self.use_ego_motion:
            self.temporal_attn_channels += 6

        self.feature_extractor = FeatureExtractor(
                 x_bound = self.cfg.LIFT.X_BOUND,
                 y_bound = self.cfg.LIFT.Y_BOUND,
                 z_bound = self.cfg.LIFT.Z_BOUND,
                 d_bound = self.cfg.LIFT.D_BOUND,
                 downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE,
                 out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS,
                 receptive_field = self.cfg.TIME_RECEPTIVE_FIELD,
                 pred_frames = self.cfg.N_FUTURE_FRAMES,
                 latent_dim = self.cfg.MODEL.DISTRIBUTION.LATENT_DIM,
                 use_depth_distribution = self.cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION,
                 model_name = self.cfg.MODEL.ENCODER.NAME,
                 img_size = self.cfg.IMAGE.FINAL_DIM,
                 )

        segformer_dims = self.cfg.MODEL.SEGFORMER.HIDDEN_SIZES
    
        ## Multiply each element by the receptive field unsing map function
        segformer_dims = list(map(lambda x: x*self.receptive_field, segformer_dims))

        segformer_config = SegformerConfig(
            image_size = self.feature_width,
            num_channels = self.cfg.TIME_RECEPTIVE_FIELD*self.temporal_attn_channels,
            num_encoder_blocks = self.cfg.MODEL.SEGFORMER.N_ENCODER_BLOCKS,
            depths = self.cfg.MODEL.SEGFORMER.DEPTHS,
            sr_ratios = self.cfg.MODEL.SEGFORMER.SEQUENCE_REDUCTION_RATIOS,
            hidden_sizes = segformer_dims,
            patch_sizes = self.cfg.MODEL.SEGFORMER.PATCH_SIZES,
            strides = self.cfg.MODEL.SEGFORMER.STRIDES,
            num_attention_heads = self.cfg.MODEL.SEGFORMER.NUM_ATTENTION_HEADS,
            mlp_ratios = self.cfg.MODEL.SEGFORMER.MLP_RATIOS,
            output_hidden_states = True,
            return_dict = True
        )
        self.segformer = SegformerModel(segformer_config)

        self.head = MultiBranchSTconv(self.cfg,
                                      in_channels=self.temporal_attn_channels)


    def forward(self, x, intrinsics, extrinsics, future_egomotion,  future_distribution_inputs=None, noise=None):
        output = {}
        start_time = time.time()

        # Image feature extraction
        x = self.feature_extractor(x, 
                                   intrinsics, extrinsics, 
                                   future_egomotion)
        
        perception_time = time.time()

        # Spatial and temporal attention
        # x = self.attention_module(x)[0]

        # Transofrmer multi-scale encoder
        b, s, c = future_egomotion.shape
        h, w = x.shape[-2:]
        future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1).expand(b, s, c, h, w)
        # At time 0, no egomotion so feed zero vector
        future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                            future_egomotions_spatial[:, :(self.receptive_field-1)]], dim=1)
        x = torch.cat([x, future_egomotions_spatial], dim=-3)

        b, t, c, h, w = x.shape
        x = x.view(b, t*c, h, w)

        x = list(self.segformer(x).hidden_states)

        for i in range(len(x)):
            b, _, h, w = x[i].shape
            x[i] = x[i].view(b, self.receptive_field, -1, h, w)


        # Shared transformer between the two different branches
        x = self.head(x)

        prediction_time = time.time()

        output['perception_time'] = perception_time - start_time
        output['prediction_time'] = prediction_time - perception_time
        output['total_time'] = output['perception_time'] + output['prediction_time']

        output = {**output, **x}

        return output
    

class PowerFormer_dualenc(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.feature_width = int((self.cfg.LIFT.X_BOUND[1] - self.cfg.LIFT.X_BOUND[0])\
                              /self.cfg.LIFT.X_BOUND[2])

        self.use_ego_motion = self.cfg.MODEL.STCONV.INPUT_EGOPOSE
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD

        self.temporal_attn_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS
        if self.use_ego_motion:
            self.temporal_attn_channels += 6

        self.feature_extractor = FeatureExtractor(
                 x_bound = self.cfg.LIFT.X_BOUND,
                 y_bound = self.cfg.LIFT.Y_BOUND,
                 z_bound = self.cfg.LIFT.Z_BOUND,
                 d_bound = self.cfg.LIFT.D_BOUND,
                 downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE,
                 out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS,
                 receptive_field = self.cfg.TIME_RECEPTIVE_FIELD,
                 pred_frames = self.cfg.N_FUTURE_FRAMES,
                 latent_dim = self.cfg.MODEL.DISTRIBUTION.LATENT_DIM,
                 use_depth_distribution = self.cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION,
                 model_name = self.cfg.MODEL.ENCODER.NAME,
                 img_size = self.cfg.IMAGE.FINAL_DIM,
                 )

        segformer_dims = self.cfg.MODEL.SEGFORMER.HIDDEN_SIZES
    
        ## Multiply each element by the receptive field unsing map function
        segformer_dims = list(map(lambda x: x*self.receptive_field, segformer_dims))

        segformer_config = SegformerConfig(
            image_size = self.feature_width,
            num_channels = self.cfg.TIME_RECEPTIVE_FIELD*self.temporal_attn_channels,
            num_encoder_blocks = self.cfg.MODEL.SEGFORMER.N_ENCODER_BLOCKS,
            depths = self.cfg.MODEL.SEGFORMER.DEPTHS,
            sr_ratios = self.cfg.MODEL.SEGFORMER.SEQUENCE_REDUCTION_RATIOS,
            hidden_sizes = segformer_dims,
            patch_sizes = self.cfg.MODEL.SEGFORMER.PATCH_SIZES,
            strides = self.cfg.MODEL.SEGFORMER.STRIDES,
            num_attention_heads = self.cfg.MODEL.SEGFORMER.NUM_ATTENTION_HEADS,
            mlp_ratios = self.cfg.MODEL.SEGFORMER.MLP_RATIOS,
            output_hidden_states = True,
            return_dict = True
        )
        
        self.segformer_segment = SegformerModel(segformer_config)
        self.segformer_flow = SegformerModel(segformer_config)

        self.head = MultiBranchSTconv_dualencoder(self.cfg,
                                      in_channels=self.temporal_attn_channels)


    def forward(self, x, intrinsics, extrinsics, future_egomotion,  future_distribution_inputs=None, noise=None):
        output = {}
        start_time = time.time()

        # Image feature extraction
        x = self.feature_extractor(x, 
                                   intrinsics, extrinsics, 
                                   future_egomotion)
        
        perception_time = time.time()

        # Spatial and temporal attention
        # x = self.attention_module(x)[0]

        # Transofrmer multi-scale encoder
        b, s, c = future_egomotion.shape
        h, w = x.shape[-2:]
        future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1).expand(b, s, c, h, w)
        # At time 0, no egomotion so feed zero vector
        future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                            future_egomotions_spatial[:, :(self.receptive_field-1)]], dim=1)
        x = torch.cat([x, future_egomotions_spatial], dim=-3)

        b, t, c, h, w = x.shape
        x = x.view(b, t*c, h, w)

        x1 = list(self.segformer_segment(x).hidden_states)
        x2 = list(self.segformer_flow(x).hidden_states)

        for i in range(len(x1)):
            b, _, h, w = x1[i].shape
            x1[i] = x1[i].view(b, self.receptive_field, -1, h, w)
            x2[i] = x2[i].view(b, self.receptive_field, -1, h, w)
                 
        # A transformer encoder for each of the two different branches
        x = self.head(x1,x2)

        prediction_time = time.time()

        output['perception_time'] = perception_time - start_time
        output['prediction_time'] = prediction_time - perception_time
        output['total_time'] = output['perception_time'] + output['prediction_time']

        output = {**output, **x}

        return output
    
    
class FullSegformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.feature_width = int((self.cfg.LIFT.X_BOUND[1] - self.cfg.LIFT.X_BOUND[0])\
                              /self.cfg.LIFT.X_BOUND[2])

        self.use_ego_motion = self.cfg.MODEL.STCONV.INPUT_EGOPOSE
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD

        self.temporal_attn_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS
        if self.use_ego_motion:
            self.temporal_attn_channels += 6

        self.feature_extractor = FeatureExtractor(
                 x_bound = self.cfg.LIFT.X_BOUND,
                 y_bound = self.cfg.LIFT.Y_BOUND,
                 z_bound = self.cfg.LIFT.Z_BOUND,
                 d_bound = self.cfg.LIFT.D_BOUND,
                 downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE,
                 out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS,
                 receptive_field = self.cfg.TIME_RECEPTIVE_FIELD,
                 pred_frames = self.cfg.N_FUTURE_FRAMES,
                 latent_dim = self.cfg.MODEL.DISTRIBUTION.LATENT_DIM,
                 use_depth_distribution = self.cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION,
                 model_name = self.cfg.MODEL.ENCODER.NAME,
                 img_size = self.cfg.IMAGE.FINAL_DIM,
                 )
                      
        segformer_out_dim = len(self.cfg.SEMANTIC_SEG.WEIGHTS)*(self.cfg.N_FUTURE_FRAMES + 2)*self.cfg.MODEL.SEGFORMER.HEAD_DIM_MULTIPLIER
       
        segformer_config = SegformerConfig(
            image_size = self.feature_width,
            num_channels = self.cfg.TIME_RECEPTIVE_FIELD*self.temporal_attn_channels,
            num_encoder_blocks = self.cfg.MODEL.SEGFORMER.N_ENCODER_BLOCKS,
            depths = self.cfg.MODEL.SEGFORMER.DEPTHS,
            sr_ratios = self.cfg.MODEL.SEGFORMER.SEQUENCE_REDUCTION_RATIOS,
            hidden_sizes = self.cfg.MODEL.SEGFORMER.HIDDEN_SIZES, # No receptive field multiplication
            patch_sizes = self.cfg.MODEL.SEGFORMER.PATCH_SIZES,
            strides = self.cfg.MODEL.SEGFORMER.STRIDES,
            num_attention_heads = self.cfg.MODEL.SEGFORMER.NUM_ATTENTION_HEADS,
            mlp_ratios = self.cfg.MODEL.SEGFORMER.MLP_RATIOS,
            output_hidden_states = True,
            return_dict = True,
            num_labels = segformer_out_dim
        )
        
        # TODO: remove last layer to avoid classification ??
        kernel = self.cfg.MODEL.SEGFORMER.HEAD_KERNEL
        stride = self.cfg.MODEL.SEGFORMER.HEAD_STRIDE
        self.segmentation_branch = SegformerForSemanticSegmentation(segformer_config)
        self.conv_seg = nn.ConvTranspose2d(segformer_out_dim,
                                           len(self.cfg.SEMANTIC_SEG.WEIGHTS)*(self.cfg.N_FUTURE_FRAMES + 2),
                                           kernel_size=kernel, stride=stride)
        
        # segformer_config.num_labels = 2
        self.flow_branch = SegformerForSemanticSegmentation(segformer_config)
        self.conv_flow = nn.ConvTranspose2d(segformer_out_dim, 2*(self.cfg.N_FUTURE_FRAMES + 2),
                                            kernel_size=kernel, stride=stride)
        
    def forward(self, x, intrinsics, extrinsics, future_egomotion,  future_distribution_inputs=None, noise=None):
        output = {}
        start_time = time.time()

        # Image feature extraction
        x = self.feature_extractor(x, 
                                   intrinsics, extrinsics, 
                                   future_egomotion)
        
        perception_time = time.time()

        # Spatial and temporal attention
        # x = self.attention_module(x)[0]

        # Transofrmer multi-scale encoder
        b, s, c = future_egomotion.shape
        h, w = x.shape[-2:]
        future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1).expand(b, s, c, h, w)
        # At time 0, no egomotion so feed zero vector
        future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                            future_egomotions_spatial[:, :(self.receptive_field-1)]], dim=1)
        x = torch.cat([x, future_egomotions_spatial], dim=-3)

        b, t, c, h, w = x.shape
        x = x.view(b, t*c, h, w)

        # Segformer directly
        seg_out= self.segmentation_branch(x).logits
        flow_out = self.flow_branch(x).logits
               
        seg_out= self.conv_seg(seg_out)
        flow_out = self.conv_flow(flow_out)      
                                
        output['segmentation'] = seg_out.view(b,self.cfg.N_FUTURE_FRAMES + 2,
                                              len(self.cfg.SEMANTIC_SEG.WEIGHTS), h, w).contiguous()
        output['instance_flow'] = flow_out.view(b,self.cfg.N_FUTURE_FRAMES + 2,
                                       len(self.cfg.SEMANTIC_SEG.WEIGHTS), h, w).contiguous()
        
        prediction_time = time.time()

        output['perception_time'] = perception_time - start_time
        output['prediction_time'] = prediction_time - perception_time
        output['total_time'] = output['perception_time'] + output['prediction_time']

        output = {**output}

        return output