# Import libraries
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import StratifiedKFold
from imgaug import augmenters as iaa

import warnings
warnings.filterwarnings("ignore")

# MaskRCNN
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

# tl;dr
TRAINING_SIZE = 512  # 1024 png or 512 jpg
AUGMENTATION = True # True or False
IMG_PER_GPU = 5
BACKBONE_ARCHITECTURE = "resnet101" # resnet 50 or resnet 101
LR = 1e-4
EPOCHS = 100

# Data directories
DATA_DIR = "/home/dairesearch/data/kaggle/data/"

TRAIN_CSV_DIR = os.path.join(DATA_DIR, "org_train.csv")

orin_df = pd.read_csv(TRAIN_CSV_DIR)
orin_df = orin_df.query('class_id != 14')

if TRAINING_SIZE == 1024:
    PREPROCESSED_TRAINING_IMAGE_FOLDER = '/home/dairesearch/data/kaggle/data/1024_png/'
    samples_df = pd.read_csv('/home/dairesearch/data/kaggle/data/1024_png_df.csv', converters={'EncodedPixels': eval, 'CategoryId': eval})
elif TRAINING_SIZE == 512:
    PREPROCESSED_TRAINING_IMAGE_FOLDER = '/home/dairesearch/data/kaggle/data/512_jpg/'
    samples_df = pd.read_csv('/home/dairesearch/data/kaggle/data/512_jpg_df.csv', converters={'EncodedPixels': eval, 'CategoryId': eval})

# # DEBUG
# samples_df = samples_df[:100]

NUM_CATS = 14
if TRAINING_SIZE == 1024:
    IMAGE_SIZE = 1024
elif TRAINING_SIZE == 512:
    IMAGE_SIZE = 512

# Create Config Normal
class DiagnosticConfig(Config):
    NAME = "Diagnostic"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class

    GPU_COUNT = 1
    IMAGES_PER_GPU = IMG_PER_GPU

    BACKBONE = BACKBONE_ARCHITECTURE

    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE

    # Try resize
    IMAGE_RESIZE_MODE = 'none'

    POST_NMS_ROIS_TRAINING = 250
    POST_NMS_ROIS_INFERENCE = 150

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 30

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # BACKBONESHAPE = (8, 16, 24, 32, 48)
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 24, 32, 48)
    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 300
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Weight decay
    # WEIGHT_DECAY = 0.0005
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 512

    STEPS_PER_EPOCH = int(len(samples_df)*0.8/IMAGES_PER_GPU)
    VALIDATION_STEPS = int(len(samples_df)/IMAGES_PER_GPU)-int(len(samples_df)*0.8/IMAGES_PER_GPU)
#
# Create Config Customize
# class DiagnosticConfig(Config):
#     NAME = "Diagnostic"
#     NUM_CLASSES = NUM_CATS + 1 # +1 for the background class
#
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = IMG_PER_GPU
#
#     BACKBONE = BACKBONE_ARCHITECTURE
#
#     IMAGE_MIN_DIM = IMAGE_SIZE
#     IMAGE_MAX_DIM = IMAGE_SIZE
#
#     # Try resize
#     IMAGE_RESIZE_MODE = "none"
#     # IMAGE_MIN_SCALE = 2.0
#
#     # POST_NMS_ROIS_TRAINING = 2000
#     # POST_NMS_ROIS_INFERENCE = 1000
#
#     # Maximum number of ground truth instances to use in one image
#     MAX_GT_INSTANCES = 50
#
#     # The strides of each layer of the FPN Pyramid. These values
#     # are based on a Resnet101 backbone.
#     BACKBONE_STRIDES = [4, 8, 16, 32, 64]
#     # BACKBONESHAPE = (8, 16, 24, 32, 48)
#     # Length of square anchor side in pixels
#     RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
#     # Percent of positive ROIs used to train classifier/mask heads
#     ROI_POSITIVE_RATIO = 0.33
#     # Max number of final detections
#     DETECTION_MAX_INSTANCES = 300
#     # Minimum probability value to accept a detected instance
#     # ROIs below this threshold are skipped
#     DETECTION_MIN_CONFIDENCE = 0.7
#
#     # Non-max suppression threshold to filter RPN proposals.
#     # You can increase this during training to generate more propsals.
#     RPN_NMS_THRESHOLD = 0.9
#
#     # TRAIN_BN = True
#
#     STEPS_PER_EPOCH = int(len(samples_df)*0.8/IMAGES_PER_GPU)
#     VALIDATION_STEPS = int(len(samples_df)/IMAGES_PER_GPU)-int(len(samples_df)*0.8/IMAGES_PER_GPU)

config = DiagnosticConfig()

category_list = ["Aortic enlargement", "Atelectasis","Calcification","Cardiomegaly","Consolidation","ILD",
                "Infiltration", "Lung opacity", "Nodule/ Mass","Other lesion","Pleural effusion",
                "Pleural thickening", "Pneumothorax","Pulmonary fibrosis"]

# Create MaskRCNN formart dataset
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
                           path= PREPROCESSED_TRAINING_IMAGE_FOLDER+str(row.image_id)+".jpg", # Check and change with jpg if 512
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

# Implement stratified k-fold
df = orin_df
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
df_folds = df[['image_id']].copy()

df_folds.loc[:, 'bbox_count'] = 1
df_folds = df_folds.groupby('image_id').count()
df_folds.loc[:, 'object_count'] = df.groupby('image_id')['class_id'].nunique()

df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['object_count'].values.astype(str),
    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
)

df_folds.loc[:, 'fold'] = 0
for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

# example with fold 0
df_folds.reset_index(inplace=True)

df_valid = pd.merge(df, df_folds[df_folds['fold'] == 0], on='image_id')
df_train = pd.merge(df, df_folds[df_folds['fold'] != 0], on='image_id')

maskrcnn_df_train = samples_df[~samples_df['image_id'].isin(df_valid['image_id'])]
maskrcnn_df_val = samples_df[~samples_df['image_id'].isin(df_train['image_id'])]

# Apply format
train_dataset = DiagnosticDataset(maskrcnn_df_train)
train_dataset.prepare()

valid_dataset = DiagnosticDataset(maskrcnn_df_val)
valid_dataset.prepare()

# AUGMENTATION IMPLEMENTATION
# Image augmentation (light but constant)
augmentation = iaa.Sequential([
    iaa.OneOf([ ## geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.04)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-2, 2),
            shear=(-1, 1),
        ),
    ]),
    iaa.OneOf([ ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])

# Load weight coco
WEIGHT_PATH = '/home/dairesearch/home/dairesearch/chest_x_ray_abnormalities_detection/MaskRCNN_implementation/weights/mask_rcnn_coco.h5'

# Create model and load pretrained weights
model = modellib.MaskRCNN(mode='training', config=config, model_dir="")

model.load_weights(WEIGHT_PATH, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

# Continue training
# model.load_weights(WEIGHT_PATH, by_name=True)

# if augmentation:
#     print('TRAINING WITH AUGMENTATION')
#     model.train(train_dataset, valid_dataset,
#                 learning_rate=LR,
#                 epochs=EPOCHS,
#                 layers='all',
#                 augmentation=augmentation)
#
# model.train(train_dataset,
#             valid_dataset,
#             learning_rate=LR,
#             epochs=EPOCHS,
#             layers='all')


# RSNA Training produce
LEARNING_RATE = 0.0001

# model.train(train_dataset, valid_dataset,
#              learning_rate=LEARNING_RATE*2,
#              epochs=1, #default 2
#              layers='heads',
#              augmentation=None)  ## no need to augment yet

model.train(train_dataset, valid_dataset,
             learning_rate=LEARNING_RATE,
             epochs=6,
             layers='all',
             augmentation=augmentation)

model.train(train_dataset, valid_dataset,
            learning_rate=LEARNING_RATE/5,
            epochs=9,
            layers='all',
            augmentation=augmentation)
model.train(train_dataset, valid_dataset,
        #learning_rate=LEARNING_RATE/5,
        learning_rate=LEARNING_RATE/10,
        epochs=12,
        layers='all',
        augmentation=augmentation)




# Plot history train/ val
history = model.keras_model.history.history

epochs = range(EPOCHS)

plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.plot(epochs, history['loss'], label="train loss")
plt.plot(epochs, history['val_loss'], label="valid loss")
plt.legend()
plt.subplot(132)
plt.plot(epochs, history['mrcnn_class_loss'], label="train class loss")
plt.plot(epochs, history['val_mrcnn_class_loss'], label="valid class loss")
plt.legend()
plt.subplot(133)
plt.plot(epochs, history['mrcnn_mask_loss'], label="train mask loss")
plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid mask loss")
plt.legend()

plt.show()