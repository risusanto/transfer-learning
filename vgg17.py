from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras import models
from keras import layers
from keras import optimizers

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

import pyprind


#root dir untuk simpan data training dan validation
base_dir = 'data'
#train and validation data
train_dir = os.path.join(base_dir, 'food-test')
validation_dir = os.path.join(base_dir, 'food-test')
#test directory
test_dir = os.path.join(base_dir, 'food-test')
datagen = ImageDataGenerator(rescale=1./255)
#banyaknya setiap data yang diretrive oleh generator sekali panggil
batch_size = 4
num_classes = 4

conv_base = VGG16(weights='imagenet',
            include_top=False,
            input_shape=(150, 150, 3))


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count,num_classes))

    generator = datagen.flow_from_directory(
                directory,
                target_size=(150, 150),
                batch_size=batch_size,
                class_mode='categorical')
    i = 0
    bar = pyprind.ProgBar(sample_count, bar_char='█', title='Load data from: '+str(directory))
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        bar.update(iterations=batch_size)
        if i * batch_size >= sample_count:
            break

    return features, labels


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))

    return model


def train_model(epochs):

    #get traing data, validation data
    train_features, train_labels = extract_features(train_dir, 24) #24, jumlah data dalam dir
    validation_features, validation_labels = extract_features(validation_dir, 24) #24, jumlah data dalam dir

    train_features = np.reshape(train_features, (len(train_labels), 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (len(validation_labels), 4 * 4 * 512))

    model = create_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    # callbacks
    base_path = 'models/'
    patience = 50
    log_file_path = base_path + 'training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                    patience=int(patience/4), verbose=1)
    early_stop = EarlyStopping('val_loss', patience=patience)
    trained_models_path = base_path + '_vgg16_'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                        save_best_only=True)
    callbacks = [model_checkpoint, csv_logger,reduce_lr, early_stop]

    model.fit(train_features, train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(validation_features, validation_labels),
                callbacks=callbacks)


if __name__ == '__main__':
    train_model(100)



