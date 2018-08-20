import os
import time
import matplotlib
import numpy as np
from skimage.io import imread

# =================================== #
#           Loading params            #
# =================================== #
DATASET_NAME = 'KITTI'
DATA_PATH = f'data_prepaired/{DATASET_NAME}' #f 'data_other/{DATASET_NAME}/training'
PREPAIRED_DATA_PATH = 'data_prepaired'
RECOMPILE_DATA = True
SAVE_RECOMPILED = False
LOAD_FIRST = 64
MODEL_NAME = 'kitti_segnet'
STATE = 'train'
RANDOM_SEED = 42
# =================================== #
#             Data params             #
# =================================== #
MASK_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
img_w = 96 #352 #1216
img_h = 96 #352
img_layers = 3
n_labels = len(MASK_COLORS)
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

# =================================== #
#      Loading images and labels      #
# =================================== #
def load_data(load_first: int = -1) -> (np.ndarray, list):
    path_img = f'{DATA_PATH}/img'
    path_lbl = f'{DATA_PATH}/gt'
    files_img = os.listdir(path_img)
    files_lbl = os.listdir(path_lbl)
    if load_first != -1:
        files_img = files_img[:load_first]
        files_lbl = files_lbl[:load_first]
    dprint('processes', 'Loading images')
    images = list(map(lambda x: imread(f'{path_img}/{x}'), files_img))
    dprint('processes', 'Loading masks')
    labels = list(map(lambda x: imread(f'{path_lbl}/{x}'), files_lbl))
    images = np.array(images) / 255.
    return images, labels

# =================================== #
#       Loading converted data        #
# =================================== #
def load_converted_data():
    file_images = f'{PREPAIRED_DATA_PATH}/{MODEL_NAME}_images.npy'
    file_labels = f'{PREPAIRED_DATA_PATH}/{MODEL_NAME}_labels.npy'

    dprint('processes', f'Loading prepaired data from {PREPAIRED_DATA_PATH}')
    images = np.load(file_images)
    dprint('processes', 'Loading lables array...')
    labels = np.load(file_labels)

    dprint('processes', 'Data loaded')
    return images, labels

# =================================== #
#   Label to neural out conversion    #
# =================================== #
def labels_conversion(labels: list) -> np.ndarray:
    loc_img_h, loc_img_w, loc_n_labels, loc_masks = img_h, img_w, n_labels, MASK_COLORS
    labels_converted = np.array([np.zeros([loc_img_h, loc_img_w, loc_n_labels])] * len(labels))
    dprint('variables', f'Numpy labels_converted created:',
           f'shape {labels_converted.shape}',
           f'memory: {labels_converted.nbytes / 1048576} MB', sep='\n\t')

    for i in range(len(labels)):
        if i % 10 == 0:
            dprint('more_proc', f'Converting {i} mask')
        for cls in range(loc_n_labels):
            labels_converted[i, :, :, cls] = labels[i] == cls

    return labels_converted

# =================================== #
#        Load and prepare data        #
# =================================== #
def data_preparation() -> (np.ndarray, np.ndarray):
    dprint('processes', 'Loading data')
    dprint('variables', f'Load first: {LOAD_FIRST}')
    images, masks = load_data(LOAD_FIRST)
    dprint('processes', 'Data loaded')

    unique_gray = set()
    for pic in masks:
        a = np.unique(pic.reshape(-1, 1))
        for el in a:
            unique_gray.add(el)
    dprint('variables', 'Unique masks pixel values:', '\t', unique_gray)

    dprint('processes', 'Converting masks...')
    labels = labels_conversion(masks)
    dprint('processes', 'Converssion done')

    labels = labels.reshape((len(images), img_h * img_w, n_labels))

    if SAVE_RECOMPILED:
        dprint('processes', 'Saving images data...')
        np.save(f'{PREPAIRED_DATA_PATH}/{MODEL_NAME}_images.npy', images)
        dprint('processes', 'Saving lables data...')
        np.save(f'{PREPAIRED_DATA_PATH}/{MODEL_NAME}_labels.npy', labels)
        dprint('processes', f'Done. Data saved to /{PREPAIRED_DATA_PATH}')

    return images, labels

if __name__ == '__main__':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    from keras import models
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    # ====== Config for Tensorflow ====== #
    tf.logging.set_verbosity(tf.logging.FATAL)
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    np.random.seed(RANDOM_SEED)

    if RECOMPILE_DATA:
        images, labels = data_preparation()
    else:
        images, labels = load_converted_data()

    dprint('processes', 'Data prepairing: Done')
    dprint('processes', '\tshapes: {}, {}'.format(images.shape, labels.shape))
    dprint('processes', '\ttypes:  {}, {}'.format(images.dtype, labels.dtype))
    dprint('processes', '\tmemory: {}, {} MB'.format(images.nbytes / 1048576, labels.nbytes / 1048576))

    if STATE == 'train':
        dprint('processes', 'Program runs in train mode')

        with open(f'models/{MODEL_NAME}.json') as model_file:
            autoencoder = models.model_from_json(model_file.read())

        # optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
        autoencoder.compile(loss="categorical_crossentropy", optimizer='ADAM', metrics=['accuracy'])

        dprint('processes', 'Model loaded and compiled')

        early_stop = EarlyStopping(monitor='val_acc', min_delta=0.0001,
                                   patience=5, verbose=1, mode='auto')
        checkpoint = ModelCheckpoint(f'models/{MODEL_NAME}.hdf5',
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='auto')

        callbacks = [early_stop, checkpoint]

        IMAGES_IN_EPOCH = 40
        CHANGE_DATA_EACH = 3
        BATCH_SIZE = 5

        # for w_crp in range(500 // CHANGE_DATA_EACH):
        history = autoencoder.fit(images, labels, batch_size=5, epochs=500,
                                  verbose=1, validation_split=0.2, callbacks=callbacks)

        autoencoder.save_weights(f'models/{MODEL_NAME}_trained.hdf5')
    elif STATE == 'test':
        dprint('processes', 'Program runs in test mode')

        autoencoder = models.load_model(f'models/{MODEL_NAME}.hdf5')

        print('Evaluating model')
        eval_result = autoencoder.evaluate(images, labels, verbose=0)
        percent = round(eval_result[1] * 100, 2)
        print(f'Model accuracy on {len(images)} images: {percent}')
    elif STATE == 'make':
        dprint('processes', 'Program runs in make mode')

        autoencoder = models.load_model(f'models/{MODEL_NAME}.hdf5')

        predict_result = autoencoder.predict(images)

        pred_imgs = np.zeros([len(images), img_w, img_h, img_layers])

        for i in range(len(predict_result)):
            pred_img = predict_result[i].reshape(img_w, img_h, n_labels)
            norm_img = np.full([img_w, img_h, 3], (0.196, 0.196, 0.196))

            for h in range(img_h):
                for w in range(img_w):
                    max_ind = np.argmax(pred_img[h][w])
                    if max_ind == 0:
                        continue
                    elif max_ind == 1:
                        norm_img[h, w, 0] = pred_img[h, w, max_ind]
                        norm_img[h, w, 1] = 0
                        norm_img[h, w, 2] = 0
                    elif max_ind == 2:
                        norm_img[h, w, 0] = 0
                        norm_img[h, w, 1] = 0
                        norm_img[h, w, 2] = pred_img[h, w, max_ind]
                    elif max_ind == 3:
                        norm_img[h, w, 0] = 0
                        norm_img[h, w, 1] = pred_img[h, w, max_ind]
                        norm_img[h, w, 2] = 0
            pred_imgs[i] = norm_img

        for i in range(20):
            img_num = np.random.randint(0, len(images))
            matplotlib.pyplot.imshow(images[img_num])
            matplotlib.pyplot.show()

            matplotlib.pyplot.imshow(pred_imgs[img_num])
            matplotlib.pyplot.show()
    else:
        print(f'Error: unknown state "{STATE}". Expected "train", "test" or "make"')
