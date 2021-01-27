# INSPECT TRAINED MODEL
# Load libraries
import os
import json
import random
from random import randint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import itertools
from tqdm import tqdm
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage import exposure
from sklearn import preprocessing
from skimage.measure import find_contours
from matplotlib.patches import Polygon

import warnings
warnings.filterwarnings("ignore")

# Load data
DATA_DIR = "/media/daitran/Data/Kaggle/VinBigData"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_CSV_DIR = os.path.join(DATA_DIR, "train.csv")
SS_CSV_DIR = os.path.join(DATA_DIR, "sample_submission.csv")

PREPROCESSED_TRAINING_IMAGE_FOLDER = '/home/daitran/Desktop/research/kaggle/VinBigData/train/512_jpg/'
resized_test_folder = '/home/daitran/Desktop/research/kaggle/VinBigData/test/'

orin_df = pd.read_csv(TRAIN_CSV_DIR)
orin_df = orin_df.query('class_id != 14')

# Load training dataframe .csv
training_df = pd.read_csv('/home/daitran/Desktop/git/chest_x_ray_abnormalities_detection/MaskRCNN_implementation/sample_df.csv', converters ={'EncodedPixels': eval, 'CategoryId': eval})
samples_df = training_df

# CONFIGURATIONS
from mrcnn.config import Config
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

NUM_CATS = 14
IMAGE_SIZE = 512

# Create Config 
class DiagnosticConfig(Config):
    NAME = "Diagnostic"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class

    GPU_COUNT = 1
    IMAGES_PER_GPU = 10 #That is the maximum with the memory available on kernels

    BACKBONE = 'resnet50'

    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE
    IMAGE_RESIZE_MODE = 'none'

    POST_NMS_ROIS_TRAINING = 250
    POST_NMS_ROIS_INFERENCE = 150
    MAX_GROUNDTRUTH_INSTANCES = 5
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    BACKBONESHAPE = (8, 16, 24, 32, 48)
    RPN_ANCHOR_SCALES = (8,16,24,32,48)
    ROI_POSITIVE_RATIO = 0.33
    DETECTION_MAX_INSTANCES = 300
    DETECTION_MIN_CONFIDENCE = 0.7


    STEPS_PER_EPOCH = int(len(samples_df)*0.8/IMAGES_PER_GPU)
    VALIDATION_STEPS = int(len(samples_df)/IMAGES_PER_GPU)-int(len(samples_df)*0.9/IMAGES_PER_GPU)

config = DiagnosticConfig()

# Create Inference Config
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Extract class names
category_list = orin_df.class_name.unique()

# Create Mask RCNN formart dataset
class DiagnosticDataset(utils.Dataset):
    def __init__(self, df):
        super().__init__(self)

        # Add classes
        for i, name in enumerate(category_list):
            self.add_class("diagnostic", i+1, name)

        # Add images
        for i, row in df.iterrows():
            self.add_image("diagnostic",
                           image_id=row.name,
                           path= PREPROCESSED_TRAINING_IMAGE_FOLDER+str(row.image_id)+".jpg",
                           labels=row['CategoryId'],
                           annotations=row['EncodedPixels'],
                           height=row['Height'], width=row['Width'],
                           img_org_id = row.image_id)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [category_list[int(x)] for x in info['labels']]

    def load_image(self, image_id):

        return cv2.imread(self.image_info[image_id]['path'])

    def load_mask(self, image_id):
        info = self.image_info[image_id]

        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)
        labels = []
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)

            annotation = [int(x) for x in annotation.split(' ')]

            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

            mask[:, :, m] = sub_mask
            labels.append(int(label)+1)
        return mask, np.array(labels)
    

# Load dataset
# Split with train = 80% samples and val = 10% and test = 10%
training_percentage = 0.8

training_set_size = int(training_percentage*len(samples_df))
validation_set_size = int((0.9-training_percentage)*len(samples_df))
test_set_size = int((0.9-training_percentage)*len(samples_df))

train_dataset = DiagnosticDataset(samples_df[:training_set_size])
train_dataset.prepare()

valid_dataset = DiagnosticDataset(samples_df[training_set_size:training_set_size+validation_set_size])
valid_dataset.prepare()

test_dataset = DiagnosticDataset(samples_df[training_set_size + validation_set_size:])
test_dataset.prepare()

# Show validation dataset information
print("Images: {}\nClasses: {}".format(len(valid_dataset.image_ids), valid_dataset.class_names))

# Call model
model = modellib.MaskRCNN(mode="inference", model_dir="",
                              config=config)
# Pretrained weight
model_path = '/home/daitran/Desktop/git/chest_x_ray_abnormalities_detection/MaskRCNN_implementation/server_weights/mask_rcnn_diagnostic_0009.h5'
model.load_weights(model_path, by_name=True)


# Display original
def display_test_result(dataset):
    image_id = random.choice(dataset.image_ids)
    print(image_id)
    print(dataset.class_names)

    # Display original Dicom

    # plot_bbox(img_id = dataset.image_info[image_id]['img_org_id'])

    # Display original in training form

    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, 
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset.class_names, figsize=(8, 8))

    # Display test prediction

    results = model.detect([original_image], verbose=1)
    r = results[0]

    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'])

display_test_result(dataset = train_dataset)

# # SUBMISSION

# def dicom2array(path, voi_lut=True, fix_monochrome=True):
#     dicom = pydicom.read_file(path)
#     # VOI LUT (if available by DICOM device) is used to
#     # transform raw DICOM data to "human-friendly" view
#     if voi_lut:
#         data = apply_voi_lut(dicom.pixel_array, dicom)
#     else:
#         data = dicom.pixel_array
#     # depending on this value, X-ray may look inverted - fix that:
#     if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
#         data = np.amax(data) - data

#     data = data - np.min(data)
#     data = data / np.max(data)
#     data = (data * 255).astype(np.uint8)

#     return data


# # Fix overlapping masks
# def refine_masks(masks, rois):
#     areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
#     mask_index = np.argsort(areas)
#     union_mask = np.zeros(masks.shape[:-1], dtype=bool)
#     for m in mask_index:
#         masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
#         union_mask = np.logical_or(masks[:, :, m], union_mask)
#     for m in range(masks.shape[-1]):
#         mask_pos = np.where(masks[:, :, m]==True)
#         if np.any(mask_pos):
#             y1, x1 = np.min(mask_pos, axis=1)
#             y2, x2 = np.max(mask_pos, axis=1)
#             rois[m, :] = [y1, x1, y2, x2]
#     return masks, rois

# def decode_rle(rle, height, width):
#     s = rle.split()
#     starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
#     starts -= 1
#     ends = starts + lengths
#     img = np.zeros(height*width, dtype=np.uint8)
#     for lo, hi in zip(starts, ends):
#         img[lo:hi] = 1
#     return img.reshape((height, width)).T

# def annotations_to_mask(annotations, height, width):
#     if isinstance(annotations, list):
#         # The annotation consists in a list of RLE codes
#         mask = np.zeros((height, width, len(annotations)))
#         for i, rle_code in enumerate(annotations):
#             mask[:, :, i] = decode_rle(rle_code, height, width)
#     else:
#         error_message = "{} is expected to be a list or str but received {}".format(annotation, type(annotation))
#         raise TypeError(error_message)
#     return mask

# def find_anomalies(dicom_image, display=False):

#     image_dimensions = dicom_image.shape

#     resized_img = cv2.resize(dicom_image, (image_size,image_size), interpolation = cv2.INTER_AREA)
#     saved_filename = resized_test_folder+"temp_image.jpg"
#     cv2.imwrite(saved_filename, resized_img) 
#     img = cv2.imread(saved_filename)

#     result = model.detect([img])
#     r = result[0]
    
#     if r['masks'].size > 0:
#         masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
#         for m in range(r['masks'].shape[-1]):
#             masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
#                                         (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
#         y_scale = image_dimensions[0]/IMAGE_SIZE
#         x_scale = image_dimensions[1]/IMAGE_SIZE
#         rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
        
#         masks, rois = refine_masks(masks, rois)
#     else:
#         masks, rois = r['masks'], r['rois']
        
#     if display:
#         visualize.display_instances(img, rois, masks, r['class_ids'], 
#                                     category_list, r['scores'],
#                                     title="prediction", figsize=(12, 12))
#     return rois, r['class_ids'], r['scores']



# # Submission function
# results = []
# test_file_list = os.listdir(TEST_DIR)
# image_size = 512

# selected_classes_dict = {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,"11":11,"12":12,"13":13}
# def keep_best_cardiomegaly_box(bbox_list, class_list, confidence_list):
#     '''
#     go through the boxes and keep only one box for 
#     cardiomegaly with the highest confidence score
#     '''
#     best_cardiomegaly_score = -1
#     best_cardiomegaly_bbox = []
#     clean_bbox_list, clean_class_list, clean_confidence_list = [],[],[]
    
#     for bbox, class_id, confidence in zip(bbox_list, class_list, confidence_list):
#         #While the class number if 3 in the dataset, it is 2 in the maskrcnn training process
#         # as I have excluded some classes
#         if class_id==2:
#             if confidence>best_cardiomegaly_score:
#                 best_cardiomegaly_score = confidence
#                 best_cardiomegaly_bbox = bbox
#         else:
#             clean_bbox_list.append(bbox)
#             clean_class_list.append(class_id)
#             clean_confidence_list.append(confidence)
            
#     if best_cardiomegaly_score>0:
#         clean_bbox_list.append(best_cardiomegaly_bbox)
#         clean_class_list.append(2)
#         clean_confidence_list.append(best_cardiomegaly_score)
        
#     return clean_bbox_list, clean_class_list, clean_confidence_list

# for image_file_name in tqdm(test_file_list):
# #     print(image_file_name)
    
#     dicom_image = dicom2array(TEST_DIR + '/' + image_file_name)
#     image_dimensions = dicom_image.shape
    
#     bbox_list, class_list, confidence_list = find_anomalies(dicom_image, display=False)
    
#     prediction_string = ""
    
#     if len(bbox_list)>0:
        
#         bbox_list, class_list, confidence_list = keep_best_cardiomegaly_box(bbox_list, class_list, confidence_list)
        
#         for bbox, class_id, confidence in zip(bbox_list, class_list, confidence_list):
            
#             class_id = next(key for key, value in selected_classes_dict.items() if value == int(class_id)-1)
#             confidence_score = str(round(confidence,3))

#             #HACK: I had to rescale the bounding box here. For some reason,
#             #It did not do it in the prediction function.
#             y_scale = image_dimensions[0]/image_size
#             x_scale = image_dimensions[1]/image_size
#             rescaled_bbox = (bbox * [y_scale, x_scale, y_scale, x_scale]).astype(int)

#             #organise the bbox into xmin, ymin, xmax, ymax
#             ymin = image_dimensions[0]-rescaled_bbox[2]
#             ymax = image_dimensions[0]-rescaled_bbox[0]
#             xmin = rescaled_bbox[1]
#             xmax = rescaled_bbox[3]
            
        
#             prediction_string += "{} {} {} {} {} {} ".format(class_id, confidence_score, xmin, ymin, xmax, ymax)
#         results.append({"image_id":image_file_name.replace(".dicom",""), "PredictionString":prediction_string.strip()})
#     else:
#         results.append({"image_id":image_file_name.replace(".dicom",""), "PredictionString":"14 1.0 0 0 1 1"})
            
# submission_df = pd.DataFrame(results)

# # print(submission_df)
# submission_df.to_csv('submission_2.csv', index=False)
