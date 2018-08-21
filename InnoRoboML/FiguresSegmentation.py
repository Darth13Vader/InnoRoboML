import os
import time
import matplotlib
import numpy as np
from skimage.io import imread

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


class Segmentator:
    def __init__(self, model_name='kitti_segnet',
                 dataset_name='KITTI',
                 dataset_path=f'data_prepaired/KITTI',
                 dataset_images_folder='img',
                 dataset_labels_folder='gt',
                 dataset_size=(128, 128, 3),
                 save_prepaired_data_to='data_prepaired',
                 load_converted_data=False,
                 save_converted_data=False,
                 random_seed=42):

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_size = dataset_size
        self.dataset_images_folder = dataset_images_folder
        self.dataset_labels_folder = dataset_labels_folder
        self.save_prepaired_data_to = save_prepaired_data_to
        self.load_converted_data = load_converted_data
        self.save_converted_data = save_converted_data

        self.random_seed = random_seed
        np.random.seed(random_seed)

        labels_files = os.listdir(f'{dataset_path}/{dataset_labels_folder}')
        self.files_number = len(labels_files)

        self.masks = -1
        self.n_labels = -1


    def load_data(self, load_from: int, load_to: int, load_first: int = -1) -> (np.ndarray, list):
        # =================================== #
        #      Loading images and labels      #
        # =================================== #
        path_img = f'{self.dataset_path}/img'
        path_lbl = f'{self.dataset_path}/gt'
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
        labels = list(map(lambda x: imread(f'{path_lbl}/{x}'), files_lbl))
        images = np.array(images) / 255.
        return images, labels


    def load_converted_data(self):
        # =================================== #
        #       Loading converted data        #
        # =================================== #
        file_images = f'{self.save_prepaired_data_to}/{self.model_name}_images.npy'
        file_labels = f'{self.save_prepaired_data_to}/{self.model_name}_labels.npy'

        dprint('processes', f'Loading prepaired data from {self.save_prepaired_data_to}')
        images = np.load(file_images)
        dprint('processes', 'Loading lables array...')
        labels = np.load(file_labels)

        dprint('processes', 'Data loaded')
        return images, labels


    def labels_conversion(self, labels: list) -> np.ndarray:
        # =================================== #
        #   Label to neural out conversion    #
        # =================================== #
        masks = set()
        for pic in masks:
            a = np.unique(pic.reshape(-1, 1))
            for el in a:
                masks.add(el)
        dprint('variables', f'Lables count: {n_labels}, values:', masks, sep='\n\t')
        masks = list(masks)
        n_labels = len(masks)

        loc_img_h, loc_img_w = img_h, img_w
        labels_converted = np.array([np.zeros([loc_img_h, loc_img_w, n_labels])] * len(labels))
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
            for cls in range(n_labels):
                labels_converted[i, :, :, cls] = labels[i] == cls

        self.masks = masks
        self.n_labels = n_labels

        return labels_converted


    def data_preparation(self) -> (np.ndarray, np.ndarray):
        # =================================== #
        #        Load and prepare data        #
        # =================================== #
        dprint('processes', 'Loading data')
        dprint('variables', f'Load by step: {LOAD_BY_STEP}')
        images, masks = load_data(*LOAD_FROM_TO, LOAD_BY_STEP)
        dprint('processes', 'Data loaded')

        dprint('processes', 'Converting masks...')
        labels = labels_conversion(masks)
        dprint('processes', 'Converssion done')

        labels = labels.reshape((len(images), img_h * img_w, n_labels))

        if SAVE_RECOMPILED:
            dprint('processes', 'Saving images data...')
            np.save(f'{self.save_prepaired_data_to}/{self.model_name}_images.npy', images)
            dprint('processes', 'Saving lables data...')
            np.save(f'{self.save_prepaired_data_to}/{self.model_name}_labels.npy', labels)
            dprint('processes', f'Done. Data saved to /{self.save_prepaired_data_to}')

        return images, labels


    def load_and_train(self, super_epoch=0):
        # =================================== #
        #        Load and train segnet        #
        # =================================== #
        if RECOMPILE_DATA:
            images, labels = data_preparation()
        else:
            images, labels = load_converted_data()

        if super_epoch == 0:
            with open(f'models/{self.model_name}.json') as model_file:
                autoencoder = models.model_from_json(model_file.read())
        else:
            autoencoder = models.load_model(f'models/{self.model_name}.hdf5')

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

        dprint('processes', f'Starting train in {super_epoch} superepoch')
        history = autoencoder.fit(images, labels, batch_size=BATCH_SIZE, epochs=1,
                                  verbose=1, validation_split=0.2, callbacks=callbacks)
        autoencoder.save_weights(f'models/{self.model_name}_trained.hdf5')
        dprint('variables', f'Super epoch {super_epoch} done', f'History: {history}', sep='\n\t')

        if super_epoch != SUPER_EPOCHS_NUMBER:
            dprint('processes', 'Deleting old data...')
            del images, labels
            dprint('processes', f'Super epoch {super_epoch} ended. Sleep a sec and start new')
            time.sleep(0.1)
        else:
            dprint('processes', f'Training ended. See results of last super epoch above')


    def run(self, state):
        if state == 'train':
            reset_next = False
            dprint('processes', 'Program runs in train mode')
            for ep in range(SUPER_EPOCHS_NUMBER):
                load_and_train(ep)
                if LOAD_FROM_TO[1] >= self.files_number - BATCH_SIZE or reset_next:
                    LOAD_FROM_TO = 0, LOAD_BY_STEP
                if LOAD_FROM_TO[1] + LOAD_BY_STEP >= self.files_number - 1:
                    LOAD_BY_STEP = LOAD_FROM_TO[1], self.files_number
                    reset_next = True
                else:
                    LOAD_FROM_TO = (LOAD_FROM_TO[1], LOAD_FROM_TO[1] + LOAD_BY_STEP)
        else:
            if RECOMPILE_DATA:
                images, labels = data_preparation()
            else:
                images, labels = load_converted_data()

            dprint('processes', 'Data prepairing: Done')
            dprint('processes', '\tshapes: {}, {}'.format(images.shape, labels.shape))
            dprint('processes', '\ttypes:  {}, {}'.format(images.dtype, labels.dtype))
            dprint('processes', '\tmemory: {}, {} MB'.format(images.nbytes / 1048576, labels.nbytes / 1048576))

            if state == 'test':
                dprint('processes', 'Program runs in test mode')

                autoencoder = models.load_model(f'models/{self.model_name}.hdf5')

                print('Evaluating model')
                eval_result = autoencoder.evaluate(images, labels, verbose=0)
                percent = round(eval_result[1] * 100, 2)
                print(f'Model accuracy on {len(images)} images: {percent}')
            elif state == 'make':
                dprint('processes', 'Program runs in make mode')

                autoencoder = models.load_model(f'models/{self.model_name}.hdf5')

                predict_result = autoencoder.predict(images)

                pred_imgs = np.zeros([len(images), *self.dataset_size])

                for i in range(len(predict_result)):
                    pred_img = predict_result[i].reshape(*self.dataset_size)
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


if __name__ == '__main__':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    from keras import models
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    # ====== Config for Tensorflow ====== #
    tf.logging.set_verbosity(tf.logging.FATAL)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.999
    set_session(tf.Session(config=config))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    # =================================== #

    # =================================== #
    #           Loading params            #
    # =================================== #
    DATASET_NAME = 'KITTI'
    DATA_PATH = f'data_prepaired/{DATASET_NAME}'  # f 'data_other/{DATASET_NAME}/training'
    PREPAIRED_DATA_PATH = 'data_prepaired'
    RECOMPILE_DATA = True
    SAVE_RECOMPILED = False
    LOAD_BY_STEP = 100  # 1000
    LOAD_FROM_TO = (0, LOAD_BY_STEP)

    # =================================== #
    #             Data params             #
    # =================================== #
    MASK_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    img_w = 128  # 352 #1216
    img_h = 128  # 352
    img_layers = 3
    n_labels = len(MASK_COLORS)
    # =================================== #
    #            Model params             #
    # =================================== #
    MODEL_NAME = 'kitti_segnet'
    STATE = 'train'
    RANDOM_SEED = 42
    SUPER_EPOCHS_NUMBER = 500
    IMAGES_IN_EPOCH = 40
    BATCH_SIZE = 5

    segmentator = Segmentator(model_name='kitti_segnet',
                              dataset_name='KITTI',
                              dataset_path=f'data_prepaired/KITTI',
                              dataset_size=(128, 128, 3),
                              save_prepaired_data_to='data_prepaired',
                              load_converted_data=False,
                              save_converted_data=False,
                              random_seed=42)
    segmentator.run(state='train',
                    superepochs_number=500,
                    data_in_superepoch=100,

                    batch_size=5)
