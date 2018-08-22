import os
import time
import matplotlib
import pickle
import numpy as np
from skimage.io import imread

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import models
from keras.callbacks import EarlyStopping, ModelCheckpoint

from additional.segnet import SegnetBuilder

# =================================== #
TIME_ST = time.time()
DEBUG_LEVELS = {
    'print all': True,
    'variables': True,
    'processes': True,
    'more_proc': True
    }


# =================================== #
#              Deb print              #
# =================================== #
def dprint(level: str, *values, sep=' ', end='\n'):
    if DEBUG_LEVELS['print all'] or DEBUG_LEVELS[level]:
        tm = str(round(time.time() - TIME_ST, 2))
        tm += '0' * (5 - len(tm))
        tm = '0' * (6 - len(tm)) + tm
        print(f'[{tm}][{level}] - ', end='')
        print(*values, sep=sep, end=end)


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


class Segmentator:
    def __init__(self, model_name='kitti_segnet',
                 dataset_name='KITTI',
                 dataset_path=f'data_prepaired/KITTI',
                 dataset_images_folder='img',
                 dataset_labels_folder='gt',
                 dataset_size=(128, 128, 3),
                 random_seed=42):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_size = dataset_size
        self.dataset_images_folder = dataset_images_folder
        self.dataset_labels_folder = dataset_labels_folder

        self.random_seed = random_seed
        np.random.seed(random_seed)

        labels_files = os.listdir(f'{dataset_path}/{dataset_labels_folder}')
        self.files_number = len(labels_files)

        self.masks = -1
        self.n_labels = -1


    def load_data(self, load_first: int = -1, load_from: int = -1, load_to: int = -1) -> (np.ndarray, list):
        # =================================== #
        #      Loading images and labels      #
        # =================================== #
        path_img = f'{self.dataset_path}/{self.dataset_images_folder}'
        path_lbl = f'{self.dataset_path}/{self.dataset_labels_folder}'
        files_img = os.listdir(path_img)
        files_lbl = os.listdir(path_lbl)
        if load_first != -1:
            try:
                files_img = files_img[load_from:load_to]
                files_lbl = files_lbl[load_from:load_to]
            except IndexError:
                files_img = files_img[load_from:]
                files_lbl = files_lbl[load_from:]
        dprint('processes', 'Loading images')
        images = list(map(lambda x: imread(f'{path_img}/{x}'), files_img))
        dprint('processes', 'Loading masks')
        labels = list(map(lambda x: imread(f'{path_lbl}/{x}')[:, :, 1], files_lbl))
        images = np.array(images) / 255.
        return images, labels


    def labels_conversion(self, labels: list) -> np.ndarray:
        # =================================== #
        #   Label to neural out conversion    #
        # =================================== #
        masks = set()
        for pic in labels:
            a = np.unique(pic.reshape(-1, 1))
            for el in a:
                masks.add(el)
        n_labels = len(masks)
        dprint('variables', f'Lables count: {n_labels}, values:', masks, sep='\n\t')
        masks = list(masks)

        img_w, img_h, _ = self.dataset_size
        labels_converted = np.array([np.zeros([img_h, img_w, n_labels])] * len(labels))
        dprint('variables', f'Numpy labels_converted created:',
               f'shape {labels_converted.shape}',
               f'memory: {labels_converted.nbytes / 1048576} MB', sep='\n\t')

        tm_start = time.time()
        prev_mask_index = 0
        for i in range(len(labels)):
            if time.time() - tm_start > 2.0:
                dprint('more_proc', f'Converting {i} mask, '
                                    f'{round((i - prev_mask_index) / (time.time() - tm_start), 2)} mask per sec')
                tm_start = time.time()
            for cls in masks:
                labels_converted[i, :, :, masks.index(cls)] = labels[i] == cls

        self.masks = masks
        self.n_labels = n_labels

        return labels_converted


    def build_model(self):
        dprint('processes', 'Building model')
        autoencoder = SegnetBuilder.build(self.model_name, *self.dataset_size, self.n_labels)
        dprint('processes', 'Model has been built')
        return autoencoder


    def compile_model(self, build_model: bool):
        if build_model:
            autoencoder = self.build_model()
        else:
            with open(f'models/{self.model_name}.json') as model_file:
                autoencoder = models.model_from_json(model_file.read())

        autoencoder.compile(loss="categorical_crossentropy", optimizer='ADAM', metrics=['accuracy'])
        dprint('processes', 'Data loaded, model ready')

        early_stop = EarlyStopping(monitor='val_acc', min_delta=0.0001,
                                   patience=10, verbose=1, mode='auto')
        checkpoint = ModelCheckpoint(f'models/{self.model_name}.hdf5',
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='auto')

        callbacks = [early_stop, checkpoint]

        return autoencoder, callbacks


    def run(self, build_model: bool, state: str, epochs: int, batch_size: int, validation_split: float, ):
        dprint('processes', 'Loading data')
        images, masks = self.load_data()
        dprint('processes', 'Data loaded')

        dprint('processes', 'Converting masks...')
        masks = self.labels_conversion(masks)
        dprint('processes', 'Converssion done')

        images, masks = shuffle_in_unison(images, masks)
        dprint('processes', 'Data shuffled')

        img_w, img_h, _ = self.dataset_size
        masks = masks.reshape((len(images), img_h * img_w, self.n_labels))

        autoencoder, callbacks = self.compile_model(build_model)
        dprint('processes', 'Model compiled and ready to use')

        dprint('processes', 'Starting train', '=' * 100, sep='\n')
        history = autoencoder.fit(images, masks, batch_size=batch_size, epochs=epochs,
                                  verbose=1, validation_split=validation_split, callbacks=callbacks)
        dprint('processes', 'Train ended, see results above', '=' * 100, sep='\n')
        autoencoder.save_weights(f'models/{self.model_name}_trained.hdf5')


if __name__ == '__main__':
    # ====== Config for Tensorflow ====== #
    tf.logging.set_verbosity(tf.logging.FATAL)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.999
    set_session(tf.Session(config=config))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    # =================================== #

    segmentator = Segmentator(model_name='figures32',
                              dataset_name='figures32',
                              dataset_path=f'data_other/figures32',
                              dataset_size=(64, 64, 3),
                              random_seed=42)
    segmentator.run(build_model=True, state='train', epochs=10, batch_size=32, validation_split=0.2)
