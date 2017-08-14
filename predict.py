from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
import pandas as pd

# Settings

test_directory = './test'

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

# base_model = applications.InceptionV3(weights='imagenet',
#                                       include_top=False,
#                                       input_shape=(img_height, img_width, 3))
# # Custom layers
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(2048, activation='relu')(x)
# # x = Dropout(0.8)(x)
# predictions = Dense(1, activation='sigmoid')(x)
# model = Model(inputs=base_model.input, outputs=predictions)
#
# model.compile(
#     loss='binary_crossentropy',
#     optimizer=optimizers.SGD(lr=0.00001,
#                              momentum=0.9,
#                              decay=0.00004),
#     metrics=['accuracy'])

model = load_model('./output/checkpoints/inceptionV3_fine_tuned_2_epoch_00_acc_0.97321.h5')

print('Model loaded.')

preds = model.predict_generator(test_generator, steps=test_samples)
preds_csv = pd.DataFrame({ 'name': range(1, test_samples),
                       'invasive': preds[0]})
preds_csv[['name', 'invasive']].to_csv('./submission.csv', index=None)
