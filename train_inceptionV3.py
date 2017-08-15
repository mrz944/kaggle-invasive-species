from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D

# Settings

train_directory = './data/train'
validation_directory = './data/validation'

img_width, img_height = 400, 300
batch_size = 8
batch_size_val = 200

train_epochs = 20
fine_tune_epochs = 50

# Data generators & augmentation
datagen = ImageDataGenerator(
    rescale=1.,
    rotation_range=10,
    width_shift_range=.1,
    height_shift_range=.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='reflect')

validgen = ImageDataGenerator(
    rescale=1.)

train_generator = datagen.flow_from_directory(
    train_directory,
    target_size=(img_height, img_width),
    class_mode='binary',
    batch_size=batch_size,
    shuffle=True)

validation_generator = validgen.flow_from_directory(
    validation_directory,
    target_size=(img_height, img_width),
    class_mode='binary',
    batch_size=batch_size_val,
    shuffle=True)

train_samples = len(train_generator.filenames)
validation_samples = len(validation_generator.filenames)

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

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=0.001),
    metrics=['accuracy'])

# train the model on the new data for a few epochs
csv_logger = CSVLogger('./output/logs/training.csv', separator=';')

tensorboard = TensorBoard(
    log_dir='./output/logs/training',
    histogram_freq=1,
    write_graph=True)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=train_epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size_val,
    verbose=1,
    callbacks=[csv_logger, tensorboard])

model.save_weights('./output/inceptionV3_20_epochs.h5')

for layer in model.layers:
    layer.trainable = True

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.adam(lr=0.0001),
    metrics=['accuracy'])

csv_logger = CSVLogger('./output/logs/fine_tuning.csv', separator=';')

checkpointer = ModelCheckpoint(
    filepath='./output/checkpoints/inceptionV3_fine_tuned_epoch_{epoch:02d}_acc_{val_acc:.5f}.h5',
    save_weights_only=True,
    save_best_only=True)

# early_stopper = EarlyStopping(patience=10)

# reduce_lr = ReduceLROnPlateau(monitor='val_acc',
#                               patience=5,
#                               verbose=1,
#                               factor=0.1,
#                               cooldown=10,
#                               min_lr=0.00001)

tensorboard = TensorBoard(
    log_dir='./output/logs/fine_tuning',
    histogram_freq=1,
    write_graph=True)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=fine_tune_epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size_val,
    verbose=1,
    callbacks=[csv_logger, checkpointer, tensorboard])

model.save_weights('./output/inceptionV3_fine_tuned_100_epochs.h5')

# serialize model to JSON
model_json = model.to_json()
with open('./output/inceptionV3_fine_tuned.json', 'w') as json_file:
    json_file.write(model_json)
