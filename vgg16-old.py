from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras import models
from keras import layers
from keras import optimizers

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model


TRAIN_DIR = "data/food-test/"
VALIDATION_DIR = "data/food-test/"
HEIGHT = 300
WIDTH = 300
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_TRAIN_DATA = 24


datagen = ImageDataGenerator(rescale=1./255)

conv_base = VGG16(weights='imagenet',
            include_top=False,
            input_shape=(HEIGHT, WIDTH, 3))


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x) 
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    return finetune_model


class_list = ["Burger", "Rendang", "Rujak","Soto"]
FC_LAYERS = [1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(conv_base, 
                                      dropout=dropout, 
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))

finetune_model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


#flow data

train_generator = datagen.flow_from_directory(TRAIN_DIR, 
                                                    target_size=(HEIGHT, WIDTH), 
                                                    batch_size=BATCH_SIZE)

validaiton_generator = datagen.flow_from_directory(VALIDATION_DIR, 
                                                    target_size=(HEIGHT, WIDTH), 
                                                    batch_size=BATCH_SIZE)
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


finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8,
                            validation_data = validaiton_generator,
                            steps_per_epoch=NUM_TRAIN_DATA // BATCH_SIZE,
                            validation_steps=NUM_TRAIN_DATA // BATCH_SIZE,
                            shuffle=True, callbacks=callbacks)