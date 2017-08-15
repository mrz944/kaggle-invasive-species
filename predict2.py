from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
import pandas as pd
# import numpy as np

# check prediction on train data
# test_set = pd.read_csv('./train_labels.csv')
test_directory = './data/test'

img_width, img_height = 400, 300

validgen = ImageDataGenerator(
    rescale=1.)

test_generator = validgen.flow_from_directory(
        test_directory,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode='binary',
        shuffle=False)

test_samples = len(test_generator.filenames)

# Loading pre-trained model and adding custom layers
base_model = applications.InceptionV3(weights='imagenet',
                                      include_top=False,
                                      input_shape=(img_height, img_width, 3))
# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.6)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights('./output/checkpoints/inceptionV3_fine_tuned_epoch_36_acc_0.99250.h5')

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.adam(lr=0.0001, decay=0.00004),
    metrics=['accuracy'])

print('Model loaded.')

preds = model.predict_generator(test_generator, test_samples)
preds_rounded = []

for pred in preds:
    if (pred > .5):
        preds_rounded.append("1")
    else:
        preds_rounded.append("0")

preds_filenames = [int(x.replace("test\\", "").replace(".jpg", "")) for x in test_generator.filenames]

data = (list(zip(preds_filenames, preds_rounded)))

df_result = pd.DataFrame(data, columns=["name", "invasive"])
df_result = df_result.sort_values("name")
df_result.index = df_result["name"]
df_result = df_result.drop(["name"], axis=1)

df_result.to_csv("./submission_test.csv", encoding="utf8", index=True)

#
# preds_test = np.zeros(test_samples, dtype=np.float)
#
# preds_test += model.predict_generator(test_generator, steps=test_samples)[:, 0]
#
# test_set['invasive'] = preds_test
# test_set.to_csv('./train_prediction.csv', index=None)
