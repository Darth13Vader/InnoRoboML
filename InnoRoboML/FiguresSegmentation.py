import os
import matplotlib
import numpy as np
from keras import models
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.io import imread

DATA_PATH = 'data_other/figures32'
MASK_COLORS = [0, 15, 45, 90]
STATE = 'make'

img_w = 32
img_h = 32
img_layers = 3
n_labels = 4


def label_map(labels):
    labels_converted = np.array([np.zeros([img_h, img_w, n_labels])] * len(labels))
    for i in range(len(labels)):
        for r in range(img_h):
            for c in range(img_w):
                labels_converted[i, r, c, MASK_COLORS.index(labels[i][r][c][0])] = 1
    return labels_converted


def prep_data():
    # img, gt = [imread(path + item[0])], np.clip(imread(path + item[1]), 0, 1)
    img = np.array([imread(f'{DATA_PATH}/img/{file}') for file in os.listdir(f'{DATA_PATH}/img')]) / 255.
    gt = [imread(f'{DATA_PATH}/gt/{file}') for file in os.listdir(f'{DATA_PATH}/gt')]
    label = label_map(gt)

    label = label.reshape((len(img), img_h * img_w, n_labels))
    print('Data prepairing: OK')
    print('\tshapes: {}, {}'.format(img.shape, label.shape))
    print('\ttypes:  {}, {}'.format(img.dtype, label.dtype))
    print('\tmemory: {}, {} MB'.format(img.nbytes / 1048576, label.nbytes / 1048576))

    return img, label


if __name__ == '__main__':
    np.random.seed(42)

    if STATE == 'train':
        with open('models/model_segnet.json') as model_file:
            autoencoder = models.model_from_json(model_file.read())

        optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
        autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        # Train model or load weights
        train_data, train_label = prep_data()

        early_stop = EarlyStopping(monitor='val_acc', min_delta=0.0001,
                                   patience=5, verbose=1, mode='auto')
        checkpoint = ModelCheckpoint('models/segnet_best.hdf5',
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='auto')
        callbacks = [early_stop, checkpoint]

        history = autoencoder.fit(train_data, train_label, batch_size=18, nb_epoch=500, verbose=1, validation_split=0.2,
                                  callbacks=callbacks)
        autoencoder.save_weights('models/model_segnet_trained.hdf5')
    elif STATE == 'test':
        print('Loading model')
        autoencoder = models.load_model('models/segnet_best.hdf5')

        print('Prepairing data')
        img, labels = prep_data()

        print('Evaluating model')
        eval_result = autoencoder.evaluate(img, labels, verbose=0)
        percent = round(eval_result[1] * 100, 2)
        print(f'Model accuracy on {len(img)} images: {percent}')
    elif STATE == 'make':
        print('Loading model')
        autoencoder = models.load_model('models/segnet_best.hdf5')

        print('Prepairing data')
        img, labels = prep_data()

        img = img[:100]
        labels = labels[:100]

        predict_result = autoencoder.predict(img)

        pred_imgs = np.zeros([len(img), 32, 32, 3])

        for i in range(len(predict_result)):
            pred_img = predict_result[i].reshape(32, 32, 4)
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

        for i in range(10):
            img_num = np.random.randint(0, len(img))
            matplotlib.pyplot.imshow(img[img_num])
            matplotlib.pyplot.show()

            matplotlib.pyplot.imshow(pred_imgs[img_num])
            matplotlib.pyplot.show()
    else:
        print(f'Error: unknown state "{STATE}". Expected "train", "test" or "make"')
