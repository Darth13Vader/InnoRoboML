import os
import time
import argparse
import pickle
import numpy as np
from skimage.io import imread, imsave

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
    def __init__(self, model_name: str, dataset_path: str,
                 dataset_images_folder: str,
                 dataset_labels_folder: str,
                 dataset_size: (int, int, int), random_seed: int = 42,
                 dataset_test_path: str = '',
                 save_predictions_to: str = ''):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.dataset_size = dataset_size
        self.dataset_images_folder = dataset_images_folder
        self.dataset_labels_folder = dataset_labels_folder

        self.dataset_test_path = dataset_test_path
        self.save_predictions_to = save_predictions_to

        self.random_seed = random_seed
        np.random.seed(random_seed)
        dprint('variables', f'Random seed set to {random_seed}')

        labels_files = os.listdir(f'{dataset_path}/{dataset_labels_folder}')
        self.files_number = len(labels_files)

        self.masks = -1
        self.n_labels = -1


    def load_data(self, load_from: int = -1, load_to: int = -1) -> (np.ndarray, list):
        # =================================== #
        #      Loading images and labels      #
        # =================================== #
        path_img = f'{self.dataset_path}/{self.dataset_images_folder}'
        path_lbl = f'{self.dataset_path}/{self.dataset_labels_folder}'
        files_img = np.array(os.listdir(path_img))
        files_lbl = np.array(os.listdir(path_lbl))

        files_img, files_lbl = shuffle_in_unison(files_img, files_lbl)
        dprint('processes', 'Files shuffled')

        if load_from != -1 and load_to != -1:
            try:
                files_img = files_img[load_from:load_to]
                files_lbl = files_lbl[load_from:load_to]
            except IndexError:
                files_img = files_img[load_from:]
                files_lbl = files_lbl[load_from:]

        dprint('processes', f'Loading {len(files_img)} images')
        images = list(map(lambda x: imread(f'{path_img}/{x}'), files_img))
        dprint('processes', f'Loading {len(files_lbl)} masks')
        labels = list(map(lambda x: imread(f'{path_lbl}/{x}')[:, :, 1], files_lbl))
        images = np.array(images) / 255.
        return images, labels


    def load_test(self):
        files_img = os.listdir(self.dataset_test_path)

        dprint('processes', f'Loading {len(files_img)} images...')
        images = list(map(lambda x: imread(f'{self.dataset_test_path}/{x}'), files_img))
        dprint('processes', f'Done')

        return np.array(images)


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
                prev_mask_index = i
            for cls in masks:
                labels_converted[i, :, :, masks.index(cls)] = labels[i] == cls

        self.masks = masks
        self.n_labels = n_labels

        return labels_converted


    def predict_conversion(self, predictions: np.ndarray) -> list:
        colors = [(255, 0, 0),  # - Red
                  (80, 227, 194),  # - Light bluish
                  (255, 118, 0),  # - Orange
                  (74, 74, 74),  # - Grey
                  (64, 119, 0),  # - Dark green
                  (0, 255, 55),  # - Green
                  (139, 87, 42),  # - Brown
                  (0, 0, 255),  # - Blue
                  (241, 234, 127),
                  (46, 74, 98)]
        colors = [np.array(x) for x in colors]
        human_masks = []

        for i in range(len(predictions)):
            outmask = predictions[i].reshape(512, 512, 9)
            conv = np.zeros((512, 512, 3))
            for w in range(512):
                for h in range(512):
                    index = int(np.argmax(outmask[w, h]))
                    conv[w, h, :] = colors[index]
            human_masks.append(conv)
        return human_masks


    def build_model(self):
        dprint('processes', 'Building model')
        autoencoder = SegnetBuilder.build(self.model_name, *self.dataset_size, self.n_labels)
        dprint('processes', 'Model has been built')
        return autoencoder


    def compile_model(self, build_model: bool, load_trained=False):
        if build_model:
            autoencoder = self.build_model()
        else:
            if load_trained:
                autoencoder = models.load_model(f'models/{self.model_name}.hdf5')
            else:
                with open(f'models/{self.model_name}.json') as model_file:
                    autoencoder = models.model_from_json(model_file.read())

        autoencoder.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
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


    def run(self, build_model: bool, state: str, epochs: int, batch_size: int, validation_split: float,
            load_from=-1, load_to=-1):
        if state == 'train':
            dprint('processes', 'Loading data')
            images, masks = self.load_data(load_from, load_to)
            dprint('processes', 'Data loaded')

            dprint('processes', 'Converting masks...')
            masks = self.labels_conversion(masks)
            dprint('processes', 'Converssion done')

            img_w, img_h, _ = self.dataset_size
            masks = masks.reshape((len(images), img_h * img_w, self.n_labels))

            autoencoder, callbacks = self.compile_model(build_model)
            dprint('processes', 'Model compiled and ready to use')

            dprint('processes', 'Starting train', '=' * 100, sep='\n')
            history = autoencoder.fit(images, masks, batch_size=batch_size, epochs=epochs,
                                      verbose=1, validation_split=validation_split, callbacks=callbacks)
            dprint('processes', 'Train ended, see results above', '=' * 100, sep='\n')
            autoencoder.save_weights(f'models/{self.model_name}_trained.hdf5')
        elif state == 'test':
            images = self.load_test()
            if build_model:
                raise ValueError('Only trained models can make predictions')
            autoencoder, callbacks = self.compile_model(False, load_trained=True)

            dprint('processes', 'Making predictions...')
            predict = autoencoder.predict(images, 1, 2)

            np.save('kek.npy', predict)
            exit(0)

            dprint('processes', 'Converting to rgb...')
            predict = self.predict_conversion(predict)
            dprint('processes', 'Done! Saving predictions results...')
            for ind, one_pic in enumerate(predict):
                imsave(f'{self.save_predictions_to}/{"0" * (2 - len(str(ind)))}{ind}.png', one_pic.astype(np.uint8))

            dprint('processes', 'Done')


# Configs #
cfg_figures32 = {'model_name':            'figures32',
                 'dataset_path':          'data_other/figures32',
                 'dataset_images_folder': 'img',
                 'dataset_labels_folder': 'gt',
                 'dataset_size':          (64, 64, 3),
                 'random_seed':           42}

load_prepaired = True
cfg_kamaz_dat = {'model_name':            'kamaz',
                 'dataset_path':          'data_prepaired/kamaz' if load_prepaired else 'data_kamaz',
                 'dataset_images_folder': 'img',
                 'dataset_labels_folder': 'masks_machine',
                 'dataset_size':          (512, 512, 3) if load_prepaired else (1280, 1024, 3),
                 'random_seed':           42,
                 'dataset_test_path':     'data_prepaired/kamaz/test',
                 'save_predictions_to':   'data_prepaired/kamaz/pred'}

if __name__ == '__main__':
    state = 'train'
    epochs = 1000
    batch_size = 8
    validation_split = 0.2

    # Еще не размеченные данные
    IGNORE_FILES = ['DJI_0052', 'DJI_0055', 'DJI_0057',
                    'DJI_0067', 'DJI_0068', 'DJI_0069',
                    'DJI_0070', 'DJI_0071', 'DJI_0072',
                    'DJI_0074', 'DJI_0077', 'DJI_0078',
                    'DJI_0082', 'DJI_0083', 'DJI_0084',
                    'DJI_0085', 'DJI_0086', 'DJI_0087',
                    'DJI_0088', 'DJI_0089', 'DJI_0093',
                    'DJI_0099', 'DJI_0103']

    # ======= Config for ArgumentParser ======= #
    parser = argparse.ArgumentParser()
    parser.add_argument('-bm', '--build_model', action='store_true')
    parser.add_argument('-s', '--state', type=str, default=state, choices=['train', 'test', 'make'])
    parser.add_argument('-e', '--epochs', type=int, default=epochs)
    parser.add_argument('-bs', '--batch_size', type=int, default=batch_size)
    parser.add_argument('-val', '--validation_split', type=float, default=validation_split)
    parser.add_argument('-from', '--load_from', type=int, default=-1)
    parser.add_argument('-to', '--load_to', type=int, default=-1)
    parser.add_argument('-seed', '--random_seed', type=int, default=42)
    args = parser.parse_args()
    # ========================================= #

    # ========= Config for Tensorflow ========= #
    # tf.logging.set_verbosity(tf.logging.FATAL)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.999
    # set_session(tf.Session(config=config))
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    # ========================================= #

    cfg_kamaz_dat['random_seed'] = args.random_seed

    segmentator = Segmentator(**cfg_kamaz_dat)

    segmentator.run(build_model=args.build_model, state=args.state,
                    epochs=args.epochs, batch_size=args.batch_size,
                    validation_split=args.validation_split,
                    load_from=args.load_from, load_to=args.load_to)
