from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

test_set = pd.read_csv('./sample_submission.csv')


def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (299, 299))
    return img

test_img = []
for img_path in tqdm(test_set['name'].iloc[:]):
    test_img.append(read_img('./data/test/test2/' + str(img_path) + '.jpg'))

test_img = np.array(test_img, np.float32) / 255

base_model = applications.InceptionV3(weights='imagenet',
                                      include_top=False,
                                      input_shape=(299, 299, 3))
# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.8)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights('./output/inceptionV3_fine_tuned_100_epochs.h5')

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.SGD(lr=0.0001,
                             momentum=0.9,
                             decay=0.00004),
    metrics=['accuracy'])

# model = load_model('./output/checkpoints/inceptionV3_fine_tuned_2_epoch_86_acc_0.98661.h5')

print('Model loaded.')

preds_test = np.zeros(len(test_img), dtype=np.float)

preds_test += model.predict(test_img)[:, 0]

test_set['invasive'] = preds_test
test_set.to_csv('./submit.csv', index=None)
