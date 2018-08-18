import os
import time
import warnings
from skimage.io import imread, imsave
from skimage.transform import resize

warnings.simplefilter('ignore')

DATA_PATH = '../data_other/KITTI'
# FOLDERS = ['training/gt', 'training/images', 'training/instance', 'training/semantic_rgb',
# 'testing/images']
FOLDERS = ['testing/images']

res_w, res_h = 1248, 384

for folder in FOLDERS:
    all_files = os.listdir(f'{DATA_PATH}/{folder}')
    tm_start = time.time()
    print(f'Starting resize in {folder} - {len(all_files)} files')
    for file in all_files:
        img = imread(f'{DATA_PATH}/{folder}/{file}')
        res_img = resize(img, (res_h, res_w))
        imsave(f'{DATA_PATH}/{folder}/{file}', res_img)
        print(f'[{round(time.time() - tm_start, 2)}] - Image {file} done')
    print('=' * 50)
    print(f'Done, eval time {round(time.time() - tm_start, 2)} sec')
    print('=' * 50)
