from keras.models import load_model
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

model = load_model('./output/checkpoints/inceptionV3_fine_tuned_2_epoch_86_acc_0.98661.h5')

print('Model loaded.')

preds_test = np.zeros(len(test_img), dtype=np.float)

preds_test += model.predict(test_img)[:, 0]

test_set['invasive'] = preds_test
test_set.to_csv('./submit.csv', index=None)
