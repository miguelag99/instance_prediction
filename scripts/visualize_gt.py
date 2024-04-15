import cv2
import os
import pdb
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision

from PIL import Image
from torchvision.io import read_image

from pyquaternion import Quaternion
from types import SimpleNamespace

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box

from tqdm import tqdm

import sys
sys.path.append('../')

from prediction.data.nuscenes_dataset import NuscenesDataset
from prediction.configs import baseline_cfg
from prediction.config import namespace_to_dict
from prediction.utils.instance import (predict_instance_segmentation,
                                       generate_gt_instance_segmentation, 
                                       convert_instance_mask_to_center_and_offset_label)
from prediction.utils.network import NormalizeInverse
from prediction.utils.geometry import (update_intrinsics, convert_egopose_to_matrix_numpy,
                                       invert_matrix_egopose_numpy, mat2pose_vec, calculate_birds_eye_view_parameters)
from prediction.utils.visualisation import (convert_figure_numpy,
                                            generate_instance_colours,
                                            make_contour, plot_instance_map)


torch.set_printoptions(sci_mode=False, linewidth=120, precision=3)

class InstancePredPlotting():
    def __init__(self, nusc_obj: NuScenes,
                 cfg: SimpleNamespace,
                 mode: str = 'val') -> None:
        """
        Class to handle the plotting of the ground truth sequences from the dataloader
        
        Args:
            nusc_obj (NuScenes): NuScenes object
            cfg (SimpleNamespace): Configuration of the dataset
            mode (str, optional): Mode of the dataset. Defaults to 'val'.
        """
        self.nusc = nusc_obj
        self.cfg = cfg
        self.mode = mode
        
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )
        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        self.CAM_NAMES = cfg.IMAGE.NAMES
        
        self.time_receptive_field = cfg.TIME_RECEPTIVE_FIELD
        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES

        self.scenes = self._get_scenes()
    
    def plot_gt_scene(self, scene_id: int,
                      vid_name: str = 'video.mp4',
                      fps: int = 1) -> None:
        """Generate a video with the ground truth sequence of a scene.

        Args:
            scene_id (int): id of the scene to be plotted.
        """
        
        # video_wr = cv2.VideoWriter(
        #     filename = vid_name,
        #     fourcc=cv2.VideoWriter_fourcc(*'mp4v'),  # Use 'XVID' para MPEG-4 codec
        #     fps = fps,
        #     frameSize = (self.frame_w, self.frame_h),
        # )
        
        # Parameters.
        T = self.sequence_length
        N = self.cfg.DATASET.N_CAMERAS
        H = self.cfg.IMAGE.FINAL_DIM[0]
        W = self.cfg.IMAGE.FINAL_DIM[1]
        
        samples = self._get_samples(scene_id)
                      
        for i in tqdm(range((len(samples)-T))):
        # for i in range(len(samples)-T):
            curr_sampl = samples[i:i+T]

            # Get the images     
            images, intrinsics, extrinsics = self.get_input_data(curr_sampl)
            instance_map, instance_dict, egopose_list, visible_instance_set = self.record_instance(curr_sampl)
            future_egomotion = self.get_future_egomotion(curr_sampl)
            
            print('---------------------------------')
            print(f'Egopose: {egopose_list}')
            
            # Refine the generated instance polygons    
            for token in instance_dict.keys():
                instance_dict[token] = self.refine_instance_poly(instance_dict[token])
            
            segmentation, instance, z_position, attribute = self.get_label(instance_dict, egopose_list)
            flow = self.get_flow_label(instance_img=instance,
                                            instance_dict=instance_dict,
                                            instance_map=instance_map,
                                            ignore_index=self.cfg.DATASET.IGNORE_INDEX)
            
            instance_centerness, instance_offset = convert_instance_mask_to_center_and_offset_label(
                instance, num_instances=len(instance_map),
                ignore_index=self.cfg.DATASET.IGNORE_INDEX,
            )
                            
            # Add batch dimension
            image = images.unsqueeze(0)
            segmentation = segmentation.unsqueeze(0)
            flow = flow.unsqueeze(0)
            centerness = instance_centerness.unsqueeze(0)
            
            device = torch.device('cuda:0')
            
            time_range = self.cfg.TIME_RECEPTIVE_FIELD
            data = {
                'segmentation': segmentation[:, time_range:].to(device),
                'instance_flow': flow[:, time_range:].to(device),
                'centerness': centerness[:, time_range:].to(device),
            }
            
            # Process ground truth
            consistent_instance_seg, matched_centers = generate_gt_instance_segmentation(
                data, compute_matched_centers=True,
                spatial_extent=(self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
            )

            first_instance_seg = consistent_instance_seg[0, 1]

            # Plot future trajectories
            unique_ids = torch.unique(first_instance_seg).cpu().long().numpy()[1:]
            instance_map = dict(zip(unique_ids, unique_ids))
            instance_colours = generate_instance_colours(instance_map)
            vis_image = plot_instance_map(first_instance_seg.cpu().numpy(), instance_map)
            trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
            for instance_id in unique_ids:
                path = matched_centers[instance_id]
                for t in range(len(path) - 1):
                    color = instance_colours[instance_id].tolist()
                    cv2.line(trajectory_img, tuple(map(int,path[t])), tuple(map(int,path[t+1])),
                            color, 4)

            # Overlay arrows
            temp_img = cv2.addWeighted(vis_image, 0.7, trajectory_img, 0.3, 0.0)
            mask = ~ np.all(trajectory_img == 0, axis=2)
            vis_image[mask] = temp_img[mask]
            
            # Plot ego pose at the center of the image with cv2 circle
            H,W = vis_image.shape[:2]
            # vis_image = cv2.circle(vis_image, (W//2, H//2), 1, (0, 0, 0), -1)
            # vis_image = cv2.rectangle(vis_image, (W//2-3, H//2-5),  (W//2+3, H//2+5), (0, 0, 0), -1)


            # Plot present RGB frames and predictions
            val_w = 4.99
            cameras = cfg.IMAGE.NAMES
            image_ratio = cfg.IMAGE.ORIGINAL_DIM[0] / cfg.IMAGE.ORIGINAL_DIM[1]
            val_h = val_w * image_ratio
            fig = plt.figure(figsize=(4 * val_w, 2 * val_h))
            width_ratios = (val_w, val_w, val_w, val_w)
            gs = mpl.gridspec.GridSpec(2, 4, width_ratios=width_ratios)
            gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

            denormalise_img = torchvision.transforms.Compose(
                (# NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                torchvision.transforms.ToPILImage(),)
            )
            # for imgi, img in enumerate(image[0, -1]):
            for imgi, img in enumerate(image[0, self.time_receptive_field]):
                ax = plt.subplot(gs[imgi // 3, imgi % 3])
                showimg = denormalise_img(img.cpu())
                if imgi > 2:
                    showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)

                plt.annotate(cameras[imgi].replace('_', ' ').replace('CAM ', ''), (0.01, 0.87), c='white',
                            xycoords='axes fraction', fontsize=14)
                plt.imshow(showimg)
                plt.axis('off')

            ax = plt.subplot(gs[:, 3])
            plt.imshow(make_contour(vis_image[::-1, ::-1]))
            plt.axis('off')

            plt.draw()
            out_frame = convert_figure_numpy(fig)
            plt.close()
            
            out_frame = Image.fromarray(out_frame)
            out_frame.save(f'plots/{self.mode}/scene_{scene_id}/prueba_{i}.png')

        
               
        # video_wr.release()

    def _get_scenes(self, version = "v1.0-mini") -> list:
            """Obtain the list of scenes in the given split.
            Args:  
                version (str, optional): Version of the dataset. Defaults to "v1.0-mini".     
            Returns:
                list: _description_
            """
            scenes = {
                "v1.0-trainval": {
                    "train": splits.train,
                    "val": splits.val,
                },
                "v1.0-test": {
                    "train": [],
                    "val": splits.test,
                },
                "v1.0-mini": {
                    "train": splits.mini_train,
                    "val": splits.mini_val,
                },
            }
            return scenes[self.cfg.DATASET.VERSION][self.mode]

    def _get_samples(self, scene_id: list) -> list:
        """Find and sort the samples in the given split by scene.
        Attributes:
            scene_id (list): id of the specified scene to be plotted.
        Returns:
            list: list of samples in the given split.
        """
        samples = [sample for sample in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [sample for sample in samples if \
            self.nusc.get('scene', sample['scene_token'])['name'] in self.scenes[scene_id]]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
         
    def get_input_data(self, samples):
        """Obtain the input image as well as the intrinsics and extrinsics parameters of the corresponding camera.

        Args:
            sample_info (dict): a dict containing the information tokens of the sample.
        """
        scale_ratio = self.cfg.IMAGE.RESIZE_SCALE
        T = self.sequence_length
        N = self.cfg.DATASET.N_CAMERAS
        H1 = self.cfg.IMAGE.ORIGINAL_HEIGHT
        W1 = self.cfg.IMAGE.ORIGINAL_WIDTH
        CAMERAS = self.cfg.IMAGE.NAMES

        images = torch.zeros((T, N, 3, H1, W1), dtype=torch.float)
        intrinsics = torch.zeros((T, N, 3, 3), dtype=torch.float)
        extrinsics = torch.zeros((T, N, 4, 4), dtype=torch.float)

        for i, sample in enumerate(samples):
            
            # From lidar egopose to world.
            lidar_sample = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
            yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
            lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
            lidar_translation = np.array(lidar_pose['translation'])[:, None]
            lidar_to_world = np.vstack([
                np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
                np.array([0, 0, 0, 1])
            ])

            for j, cam in enumerate(CAMERAS):

                camera_sample = self.nusc.get('sample_data', sample['data'][cam])
                # From world to ego pose.
                car_egopose = self.nusc.get('ego_pose', camera_sample['ego_pose_token'])
                egopose_rotation = Quaternion(car_egopose['rotation']).inverse
                egopose_translation = -np.array(car_egopose['translation'])[:, None]
                world_to_car_egopose = np.vstack([
                    np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
                    np.array([0, 0, 0, 1])
                ])
                # From egopose to sensor
                sensor_sample = self.nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
                intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
                sensor_rotation = Quaternion(sensor_sample['rotation'])
                sensor_translation = np.array(sensor_sample['translation'])[:, None]
                car_egopose_to_sensor = np.vstack([
                    np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                    np.array([0, 0, 0, 1])
                ])
                car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

                # Combine all the transformation.
                # From sensor to lidar.
                lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
                sensor_to_lidar = torch.from_numpy(np.linalg.inv(lidar_to_sensor)).float()

                # Load image.
                image_filename = os.path.join(self.nusc.dataroot, camera_sample['filename'])
                images[i,j,:,:] = read_image(image_filename).float()/255.0

                # # Normalize image.
                # image = self.normalise_image(image)
                intrinsics[i,j,:,:] = intrinsic
                extrinsics[i,j,:,:] = sensor_to_lidar               

        return images, intrinsics, extrinsics        

    def record_instance(self, samples):
        """
        Record information about each visible instance in the sequence, assigning a unique ID to each instance.
        
        Args:
            sample_info_indices (list): list of indices of the samples in the sequence.
        Returns:
            instance_map (dict): a dictionary mapping each instance token to a unique ID.
            instance_dict (dict): a dictionary containing detailed information about each instance token.
            egopose_list (np.ndarray): array of egoposes of the sequence.
            visible_instance_set (set): set of instance tokens that are visible in the sequence.
        """

        instance_map = {}
        instance_dict = {}
        egopose_list = np.empty(len(samples), dtype=object)
        visible_instance_set = set()

        for i, sample_info in enumerate(samples):
           
            egopose = self.nusc.get('ego_pose',
                        self.nusc.get('sample_data', sample_info['data']['LIDAR_TOP'])['ego_pose_token'])
            traslation = -np.array(egopose['translation'])
            yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
            rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
            
            egopose_list[i] = [traslation, rotation]

            for annotation_token in sample_info['anns']:

                annotation = self.nusc.get('sample_annotation', annotation_token)

                # Filter out all non vehicle instances
                if 'vehicle' not in annotation['category_name']:
                    continue
                # Filter out invisible vehicles
                if self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1 and annotation['instance_token'] not in visible_instance_set:
                    continue
                # Filter out vehicles that have not been seen in the past
                if i >= self.cfg.TIME_RECEPTIVE_FIELD and annotation['instance_token'] not in visible_instance_set:
                    continue
                visible_instance_set.add(annotation['instance_token'])

                if annotation['instance_token'] not in instance_map:
                    instance_map[annotation['instance_token']] = len(instance_map) + 1
                instance_id = instance_map[annotation['instance_token']]
                instance_attribute = int(annotation['visibility_token'])

                if annotation['instance_token'] not in instance_dict:
                    # For the first occurrence of an instance
                    instance_dict[annotation['instance_token']] = {
                        'timestep': [i],
                        'translation': [annotation['translation']],
                        'rotation': [annotation['rotation']],
                        'size': annotation['size'],
                        'instance_id': instance_id,
                        'attribute_label': [instance_attribute],
                    }
                else:
                    # For the instance that have appeared before
                    instance_dict[annotation['instance_token']]['timestep'].append(i)
                    instance_dict[annotation['instance_token']]['translation'].append(annotation['translation'])
                    instance_dict[annotation['instance_token']]['rotation'].append(annotation['rotation'])
                    instance_dict[annotation['instance_token']]['attribute_label'].append(instance_attribute)

        return instance_map, instance_dict, egopose_list, visible_instance_set
    

    def get_future_egomotion(self, samples) -> np.ndarray:
        """
        Obtain the egomotion in the corresponding sequence.

        Args:
            sample_info_indices (list): list of indices of the samples in the sequence.
        Returns:
            future_egomotion (np.ndarray): array with egomotion of the sequence.
        """

        future_egomotion = np.empty((len(samples),6))

        for i, sample_info in enumerate(samples):

            rec_t0 = sample_info

            future_ego = np.eye(4, dtype=float)

            if i < len(samples) - 1:
                rec_t1 = samples[i + 1]

                if rec_t0['scene_token'] == rec_t1['scene_token']:
                    egopose_t0 = self.nusc.get(
                        'ego_pose', self.nusc.get('sample_data',
                                                  rec_t0['data']['LIDAR_TOP'])['ego_pose_token']
                    )
                    egopose_t1 = self.nusc.get(
                        'ego_pose', self.nusc.get('sample_data',
                                                  rec_t1['data']['LIDAR_TOP'])['ego_pose_token']
                    )
                    egopose_t0 = convert_egopose_to_matrix_numpy(egopose_t0)
                    egopose_t1 = convert_egopose_to_matrix_numpy(egopose_t1)

                    future_ego = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
                    future_ego[3,:3] = 0.0
                    future_ego[3,3] = 1.0


            future_ego = torch.from_numpy(future_ego)
            future_ego = mat2pose_vec(future_ego).unsqueeze(0)
            future_egomotion[i,:] = future_ego
            
        return torch.from_numpy(future_egomotion).float()
    
    
    def refine_instance_poly(self, instance):
        """
        Fix the missing frames and disturbances of ground truth caused by noise.
        """
        pointer = 1
        for i in range(instance['timestep'][0] + 1, self.sequence_length):
            # Fill in the missing frames
            if i not in instance['timestep']:
                instance['timestep'].insert(pointer, i)
                instance['translation'].insert(pointer, instance['translation'][pointer-1])
                instance['rotation'].insert(pointer, instance['rotation'][pointer-1])
                instance['attribute_label'].insert(pointer, instance['attribute_label'][pointer-1])
                pointer += 1
                continue
            
            # Eliminate observation disturbances
            if self._check_consistency(instance['translation'][pointer], instance['translation'][pointer-1]):
                instance['translation'][pointer] = instance['translation'][pointer-1]
                instance['rotation'][pointer] = instance['rotation'][pointer-1]
                instance['attribute_label'][pointer] = instance['attribute_label'][pointer-1]
            pointer += 1
        
        return instance
    
    def _get_poly_region_in_image(self, instance_annotation, present_egopose):
        """
        Obtain the bounding box polygon of the instance.
        """
                
        present_ego_translation, present_ego_rotation = present_egopose

        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(present_ego_translation)
        box.rotate(present_ego_rotation)
        pts = box.bottom_corners()[:2].T

        if self.cfg.LIFT.X_BOUND[0] <= pts.min(axis=0)[0] and pts.max(axis=0)[0] <= self.cfg.LIFT.X_BOUND[1] \
            and self.cfg.LIFT.Y_BOUND[0] <= pts.min(axis=0)[1] and pts.max(axis=0)[1] <= self.cfg.LIFT.Y_BOUND[1]:
            
            pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            z = box.bottom_corners()[2, 0]
            return pts, z
        else:
            return None, None
    
    def get_label(self, instance_dict, egopose_list):
        """
        Generate labels for semantic segmentation, instance segmentation, z position, attribute from the raw data of nuScenes.
        """

        visible_instance_set = set()

        segmentation = np.zeros((self.sequence_length,self.bev_dimension[0], self.bev_dimension[1]))
        instance = np.zeros((self.sequence_length,self.bev_dimension[0], self.bev_dimension[1]))
        z_position = np.zeros((self.sequence_length,self.bev_dimension[0], self.bev_dimension[1]))
        attribute_label = np.zeros((self.sequence_length,self.bev_dimension[0], self.bev_dimension[1]))

        for i in range(self.sequence_length):
            timestep = i

            for instance_token, instance_annotation in instance_dict.items():
                if timestep not in instance_annotation['timestep']:
                    continue
                pointer = instance_annotation['timestep'].index(timestep)
                annotation = {
                    'translation': instance_annotation['translation'][pointer],
                    'rotation': instance_annotation['rotation'][pointer],
                    'size': instance_annotation['size'],
                }
                
                print(annotation)

                poly_region, z = self._get_poly_region_in_image(annotation, egopose_list[self.cfg.TIME_RECEPTIVE_FIELD - 1]) 
                
                if isinstance(poly_region, np.ndarray):
                    if i >= self.cfg.TIME_RECEPTIVE_FIELD and instance_token not in visible_instance_set:
                        continue
                    visible_instance_set.add(instance_token)

                    cv2.fillPoly(instance[i], [poly_region], instance_annotation['instance_id'])
                    cv2.fillPoly(segmentation[i], [poly_region], 1.0)
                    cv2.fillPoly(z_position[i], [poly_region], z)
                    cv2.fillPoly(attribute_label[i], [poly_region], instance_annotation['attribute_label'][pointer]) 
                
            ego_info = {
                'translation': -egopose_list[timestep][0],
                # Quaternion of zeros
                'rotation': Quaternion([0, 0, 0, 0]),
                'size': [2.0, 4.5, 1.4],
            }      
            # print(ego_info)
            poly_ego,_ = self._get_poly_region_in_image(ego_info, egopose_list[self.cfg.TIME_RECEPTIVE_FIELD - 1])

            if isinstance(poly_ego, np.ndarray):
                cv2.fillPoly(segmentation[i], [poly_ego], 1.0)

        # Plot ego pose hystory
        for pose in egopose_list[:self.cfg.TIME_RECEPTIVE_FIELD]:
            print(pose)
            annotation = {
                'translation': pose[0],
                'rotation': pose[1],
                'size': [2.0, 2.0, 2.0],
            }
            poly_region, _ = self._get_poly_region_in_image(annotation, pose) 
            
            if isinstance(poly_region, np.ndarray):
                cv2.fillPoly(segmentation[:], [poly_region], 1.0)
    

        segmentation = torch.from_numpy(np.expand_dims(segmentation,1)).long()
        instance = torch.from_numpy(instance).long()
        z_position = torch.from_numpy(np.expand_dims(z_position,1)).float()
        attribute_label = torch.from_numpy(np.expand_dims(attribute_label,1)).long()

        return segmentation, instance, z_position, attribute_label      

    @staticmethod
    def _check_consistency(translation, prev_translation, threshold=1.0):
        """
        Check for significant displacement of the instance adjacent moments.
        """
        x, y = translation[:2]
        prev_x, prev_y = prev_translation[:2]

        if abs(x - prev_x) > threshold or abs(y - prev_y) > threshold:
            return False
        return True
    
    @staticmethod
    def generate_flow(flow, instance_img, instance, instance_id):
        """
        Generate ground truth for the flow of each instance based on instance segmentation.
        """        
        _, h, w = instance_img.shape
        x, y = torch.meshgrid(torch.arange(h, dtype=torch.float), torch.arange(w, dtype=torch.float), indexing='ij')
        grid = torch.stack((x, y), dim=0)

        # Set the first frame
        instance_mask = (instance_img[0] == instance_id)
        flow[0, 1, instance_mask] = grid[0, instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
        flow[0, 0, instance_mask] = grid[1, instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]

        for i, timestep in enumerate(instance['timestep']):
            if i == 0:
                continue

            instance_mask = (instance_img[timestep] == instance_id)
            prev_instance_mask = (instance_img[timestep-1] == instance_id)
            if instance_mask.sum() == 0 or prev_instance_mask.sum() == 0:
                continue

            # Centripetal backward flow is defined as displacement vector from each foreground pixel at time t to the object center of the associated instance identity at time t−1
            flow[timestep, 1, instance_mask] = grid[0, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
            flow[timestep, 0, instance_mask] = grid[1, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]

        return flow
    
    def get_flow_label(self, instance_img, instance_dict, instance_map, ignore_index=255):
        """
        Generate the global map of the flow ground truth.
        """
        seq_len, h, w = instance_img.shape
        flow = ignore_index * torch.ones(seq_len, 2, h, w)

        for token, instance in instance_dict.items():
            flow = self.generate_flow(flow, instance_img, instance, instance_map[token])
        return flow




if __name__ == '__main__':
    
    cfg = baseline_cfg
    cfg.DATASET.DATAROOT = '/home/perception/Datasets/nuscenes/mini'
    cfg.DATASET.VERSION = 'v1.0-mini'

    mode = 'val'
   
    nusc = NuScenes(
        version=cfg.DATASET.VERSION,
        dataroot=cfg.DATASET.DATAROOT,
        verbose=True
    )
    
    plot_obj = InstancePredPlotting(nusc, cfg)
    
    scene_id = 0
    save_path = os.path.join('plots', f'{mode}', f'scene_{scene_id}')
       
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    print(plot_obj.scenes)
    
    import pdb; pdb.set_trace()
    
    plot_obj.plot_gt_scene(scene_id)
        
    # plot_obj.plot_gt_seq(idx = 0)

    ## Mirar el output de la red

