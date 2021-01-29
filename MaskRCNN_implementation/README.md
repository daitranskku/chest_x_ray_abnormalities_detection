# MASK R-CNN Implementation

## Convert to 512 jpg
After converting to 512 jpg files, we received 4395 images in jpg format and the .csv files as follow
![org](figures/512_org.png)

To implement MaskRCNN, the mask coordinate need to be converted to the encoded pixel as follow

![encoded](figures/512_encoded.png)

File [inspect_data](inspect_data.ipynb) compare the encoded pixel with Mask RCNN formart and original dicom format.

## Training 

#### Split samples with train = 80%, val = 10% and test = 10%
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

## Inspect trained model

[model_evaluation](model_evaluation.ipynb) performed model evaluation with test dataset

## Sample submission
- Using pretrained model at epoch 27 
- Using binary classication model for second step filter
$$-> LB: 0.125$$

## Some notes for improving model:
1. Develop binary classication model: Normal VS Abnormal
2. Tuning model parameter and change backbone 
3. Check bbox fussion for training and testing step
4. Using stratified k-fold
5. TTA 