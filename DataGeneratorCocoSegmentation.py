
import random
import numpy
import torch
import os, sys
from DLStudio import *
import copy
from pycocotools.coco import COCO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as tvt
import time
from torch.utils.data import Dataset, DataLoader
import cv2
import json

# In[ ]:


#to extract the required images
# Path to COCO dataset annotations
train_annotation_path = 'COCO/2017/annotations/instances_train2017.json' #location of train annotation
val_annotation_path = 'COCO/2017/annotations/instances_val2017.json' #location of val annotation
train_image_path= 'COCO/2017/train2017' #location of train images
val_image_path= 'COCO/2017/val2017' #location of val images


# Categories of interest
categories_of_interest = ['cake', 'dog', 'motorcycle'] #list of categories by name
min_object_area = 40000
target_image_size = (256, 256)


def resize_and_scale_bbox(image, bbox, segmentation, target_size):
    # Resize image
    resized_image = cv2.resize(image, target_size)

    # Calculate scaling factors for bounding box coordinates
    scale_x = target_size[0] / image.shape[1]
    scale_y = target_size[1] / image.shape[0]

    # Resize bounding box coordinates
    x, y, w, h = bbox
    resized_bbox = [int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)]

    # Resize segmentation masks
    resized_segmentation = []
    for seg in segmentation:
        # Convert segmentation polygon to numpy array
        seg_np = numpy.array(seg).reshape(-1, 2)
        # Scale segmentation points
        resized_seg_np = (seg_np * [scale_x, scale_y]).astype(int)
        # Reshape to original format
        resized_seg = resized_seg_np.reshape(-1).tolist()
        resized_segmentation.append(resized_seg)

    return resized_image, resized_bbox, resized_segmentation



# Load COCO dataset
coco_train = COCO(train_annotation_path)
coco_val = COCO(val_annotation_path)


def get_category_id(category_name):
    cat_ids = coco_train.getCatIds(catNms=[category_name])
    if cat_ids:
        return cat_ids[0]
    else:
        return None
category_ids_of_interest= [get_category_id(category) for category in categories_of_interest]
category_id_to_class_map= {key:value for (key, value) in zip(category_ids_of_interest, categories_of_interest) }
category_id_to_classid_map= {key:value for (key, value) in zip(category_ids_of_interest, [0, 1, 2]) }


# In[ ]:


train_single_object_data = []
train_repeat_check= []
for img_id in coco_train.imgs:
    ann_ids = coco_train.getAnnIds(imgIds=img_id)
    anns = coco_train.loadAnns(ann_ids)

    for ann in anns:
        if ann['category_id'] in category_ids_of_interest and ann['area'] > min_object_area:
            image_path = os.path.join(train_image_path, coco_train.imgs[img_id]['file_name'])
            if image_path in train_repeat_check:
              continue
            train_repeat_check.append(image_path)
            image = cv2.imread(image_path)
            segmentation= ann['segmentation']
            if(type(segmentation)!=list):
              continue
            if image.any()==None:
              continue
            resized_image, resized_bbox, resized_segmentation = resize_and_scale_bbox(image, ann['bbox'], segmentation, target_image_size)
            train_single_object_data.append({
                'image_name': coco_train.imgs[img_id]['file_name'],
                'image_size': target_image_size,
                'resized_image': resized_image,  # Add resized image
                'bbox': resized_bbox,
                'category_id': ann['category_id'],
                'class_name': category_id_to_class_map[ann['category_id']],
                'class_id' : category_id_to_classid_map[ann['category_id']],
                'segmentation': resized_segmentation
            })


# Filter and process images for testing set (Single Object)
test_single_object_data = []
test_repeat_check=[]
for img_id in coco_val.imgs:
    ann_ids = coco_val.getAnnIds(imgIds=img_id)
    anns = coco_val.loadAnns(ann_ids)
    for ann in anns:
        if ann['category_id'] in category_ids_of_interest and ann['area'] > min_object_area:

            image_path = os.path.join(val_image_path, coco_val.imgs[img_id]['file_name'])
            if image_path in test_repeat_check:
              continue
            test_repeat_check.append(image_path)
            image = cv2.imread(image_path)
            segmentation= ann['segmentation']
            if(type(segmentation)!=list):
              continue
            if image.any()==None:
              continue
            resized_image, resized_bbox, resized_segmentation = resize_and_scale_bbox(image, ann['bbox'], segmentation, target_image_size)
            test_single_object_data.append({
                'image_name': coco_val.imgs[img_id]['file_name'],
                'image_size': target_image_size,
                'resized_image': resized_image,  # Add resized image
                'bbox': resized_bbox,
                'category_id': ann['category_id'],
                'class_name': category_id_to_class_map[ann['category_id']],
                'class_id' : category_id_to_classid_map[ann['category_id']],
                'segmentation': resized_segmentation
            })


Save the processed images and annotations to disk for both training and testing sets (Single Object)
train_single_object_output_dir = '/Extracted_CocoLOAD/train_single_object'
test_single_object_output_dir = '/Extracted_CocoLOAD/validation_single_object'






In[ ]:


Save the processed images and annotations to disk for both training and testing sets (Single Object)
train_single_object_output_dir = 'SegmentationData/train_single_object'
test_single_object_output_dir = 'SegmentationData/validation_single_object'

# Create output directories if they don't exist
os.makedirs(train_single_object_output_dir, exist_ok=True)
os.makedirs(test_single_object_output_dir, exist_ok=True)

# Save training images and annotations for single objects
saved_image_train_single_object = []
for i, data in enumerate(train_single_object_data):
    if data in saved_image_train_single_object:
        continue
    saved_image_train_single_object.append(data)
    image_name = f"train_single_object_{i}.jpg"
    cv2.imwrite(os.path.join(train_single_object_output_dir, image_name), data['resized_image'])
    with open(os.path.join(train_single_object_output_dir, f"train_single_object_{i}.json"), 'w') as f:
        json.dump({
            'image_name': image_name,
            'image_size': target_image_size,
            'bbox': data['bbox'],
            'category_id': data['category_id'],
            'class_name': data['class_name'],
            'class_id': data['class_id'],
            'segmentation': data['segmentation'],
            'num_objects': 1,
        }, f)

# Save testing images and annotations for single objects
saved_image_test_single_object = []  # To make sure that the same image is not repeated
for i, data in enumerate(test_single_object_data):
    if data in saved_image_test_single_object:
        continue
    saved_image_test_single_object.append(data)
    image_name = f"test_single_object_{i}.jpg"
    cv2.imwrite(os.path.join(test_single_object_output_dir, image_name), data['resized_image'])
    with open(os.path.join(test_single_object_output_dir, f"test_single_object_{i}.json"), 'w') as f:
        json.dump({
            'image_name': image_name,
            'image_size': target_image_size,
            'bbox': data['bbox'],
            'category_id': data['category_id'],
            'class_name': data['class_name'],
            'class_id': data['class_id'],
            'segmentation': data['segmentation'],
            'num_objects': 1,
        },f)
