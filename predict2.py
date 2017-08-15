from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
import pandas as pd
import numpy as np

test_set = pd.read_csv('./sample_submission.csv')
test_directory = './data/test'

img_width, img_height = 299, 299
test_samples = 1531

test_generator = ImageDataGenerator().flow_from_directory(
    test_directory,
    target_size=(img_height, img_width),
    color_mode='rgb',
    class_mode='binary',
    batch_size=1,
    shuffle=False)

# Loading pre-trained model and adding custom layers

base_model = applications.InceptionV3(weights='imagenet',
                                      include_top=False,
                                      input_shape=(img_height, img_width, 3))
# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.6)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights('./output/checkpoints/inceptionV3_fine_tuned_adam_epoch_59_acc_1.00000.h5')

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.adam(lr=0.0001, decay=0.00004),
    metrics=['accuracy'])

print('Model loaded.')

preds_test = np.zeros(test_samples, dtype=np.float)

preds_test += model.predict_generator(test_generator, steps=test_samples)[:, 0]

test_set['invasive'] = preds_test
test_set.to_csv('./submission_adam.csv', index=None)
