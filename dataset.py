import os
import glob
import ntpath
import pandas as pd
import shutil
from random import shuffle

# DATA
# https://www.kaggle.com/c/invasive-species-monitoring/download/train_labels.csv.zip
# https://www.kaggle.com/c/invasive-species-monitoring/download/train.7z

VAL_PERCENT = 5

train_labels = pd.read_csv('./train_labels.csv', index_col=[0])
train_paths = glob.glob('./train/*.jpg')

print('Found %d images.') % len(train_paths)

invasive_images = []
non_invasive_images = []

for image in train_paths:
    filename = ntpath.basename(image).split('.')[0]
    image_name = ntpath.basename(image).split('.')[0]
    invasive = train_labels.loc[int(image_name)][0]
    if invasive == 1:
        invasive_images.append(image)
    else:
        non_invasive_images.append(image)

print('Inavsive: %d') % len(invasive_images)
print('Non-inavsive %d') % len(non_invasive_images)

shuffle(invasive_images)
shuffle(non_invasive_images)

nb_invasive_val = len(invasive_images) // (100 // VAL_PERCENT)
nb_non_invasive_val = len(non_invasive_images) // (100 // VAL_PERCENT)

# TRAIN DATA
print('TRAIN DATA:')
print('Invasive: %d') % (len(invasive_images) - nb_invasive_val)
print('Non-invasive %d') % (len(non_invasive_images) - nb_non_invasive_val)
for image in invasive_images[nb_invasive_val:]:
    shutil.copy2(image, os.path.join('data/train/1'))
for image in non_invasive_images[nb_non_invasive_val:]:
    shutil.copy2(image, os.path.join('data/train/0'))

# VALIDATION DATA
print('VALIDATION DATA:')
print('Invasive: %d') % nb_invasive_val
print('Non-invasive %d') % nb_non_invasive_val
for image in invasive_images[:nb_invasive_val]:
    shutil.copy2(image, os.path.join('data/validation/1'))
for image in non_invasive_images[:nb_non_invasive_val]:
    shutil.copy2(image, os.path.join('data/validation/0'))
