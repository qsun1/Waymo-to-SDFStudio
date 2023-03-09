import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from tqdm import tqdm
from PIL import Image
# save the images 
FILENAME = 'segment-54293441958058219_2335_200_2355_200_with_camera_labels.tfrecord'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')


# 对一个json写入meta.json
meta_dict = {}
meta_dict['camera_model'] = 'OPENCV'
meta_dict['height'] = 1280
meta_dict['width'] = 1920
meta_dict['has_mono_prior'] = True
meta_dict['pairs'] = None
meta_dict['worldtogt'] = [
      [1, 0, 0, 0], 
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ] # according to sdfstudio

meta_dict['scene_box'] = {
      'aabb': [
          [-1, -1, -1], # aabb for the bbox
          [1, 1, 1],
        ],
      'near': 0.5, # near plane for each image
      'far': 4.5, # far plane for each image
      'radius': 1.0, # radius of ROI region in scene
      'collider_type': 'near_far',
      # collider_type can be "near_far", "box", "sphere",
      # it indicates how do we determine the near and far for each ray
      # 1. near_far means we use the same near and far value for each ray
      # 2. box means we compute the intersection with the bounding box
      # 3. sphere means we compute the intersection with the sphere
    }

meta_dict['frames'] = []
total_count = 0
# get translation matrix statistics, translate then s
def get_translation_stat(dataset):
    translations = []
    for data in tqdm(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        vehicle2global = np.asarray(frame.pose.transform).reshape(4, 4)
        for i in range(len(frame.context.camera_calibrations)):
            cam2vehicle = np.asarray(frame.context.camera_calibrations[i].extrinsic.transform).reshape(4, 4) # camera2vehicle
            lidar2vehicle = frame.context.laser_calibrations[i].extrinsic.transform
            cam2world = np.matmul(vehicle2global, cam2vehicle)
            translations.append(cam2world[:3, 3]) # 3
    
    translations = np.stack(translations) # N, 3
    return translations.min(0), translations.max(0) # 3, 3

min_trans, max_trans = get_translation_stat(dataset)
print(min_trans, max_trans)
scale = 2 # 
axs_transform = np.asarray([[0, 0, 1, 0],
                            [-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 0, 1]])
for data in tqdm(dataset):
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))

    vehicle2global = np.asarray(frame.pose.transform).reshape(4, 4)
    for i in range(len(frame.context.camera_calibrations)):
        frame_info = {}
        img = tf.image.decode_jpeg(frame.images[i].image).numpy()
        frame_info['rgb_path'] = f'images/{total_count}.png'
        plt.imsave(f'images/{total_count}.png', img)

        
        intrinsic = frame.context.camera_calibrations[i].intrinsic # 9
        frame_info['intrinsics'] = np.array([[intrinsic[0], 0, intrinsic[2], 0],
                                [0, intrinsic[1], intrinsic[3], 0],
                                [0, 0,                       1, 0],
                                [0, 0,                       0, 1]]).tolist()

        cam2vehicle = np.asarray(frame.context.camera_calibrations[i].extrinsic.transform).reshape(4, 4) # camera2vehicle  

        lidar2vehicle = frame.context.laser_calibrations[i].extrinsic.transform
        cam2world = np.matmul(np.matmul(vehicle2global, cam2vehicle), axs_transform)
        traslations = cam2world[:3, 3] # 3
        traslations = (traslations - min_trans) /  (max_trans - min_trans)  * scale - scale/2# translation to [-scale/2, scale/2]
        # print(traslations)
        cam2world[:3, 3] = traslations

        frame_info['camtoworld'] = cam2world.tolist()

        frame_info['mono_depth_path'] = None
        frame_info['mono_normal_path'] = None

        total_count += 1
        # print(total_count)

        if Image.open(frame_info['rgb_path']).size != (1920, 1280):
            continue
        meta_dict['frames'].append(frame_info)

with open("meta_data.json", "w+") as f:
    json.dump(meta_dict, f)

print(frame.context)