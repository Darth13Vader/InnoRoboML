from skimage.io import imread, imsave
import numpy as np
import os

DATA_PATH = '../data_other/KITTI/training'
crp_w = 1242, 1216
crp_h = 375, 352

def crop():
    for folder in os.listdir(DATA_PATH):
        print(folder)
        for file in os.listdir(f'{DATA_PATH}/{folder}'):
            print('\t', file)
            img = imread(f'{DATA_PATH}/{folder}/{file}')
            img = img[:crp_h[1], :crp_w[1]]
            imsave(f'{DATA_PATH}/{folder}/{file}', img)

    print('Done')

un = set()
for file in os.listdir(f'{DATA_PATH}/gt'):
    a = np.unique(imread(f'{DATA_PATH}/gt/{file}').reshape(-1, 1))
    for el in a:
        un.add(el)
print(un)
