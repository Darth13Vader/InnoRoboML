import os
import time
import math
import warnings
import matplotlib.pyplot as plt
from skimage.io import imread, imsave, imshow
from FiguresSegmentation import DATASET_NAME, DATA_PATH
from FiguresSegmentation import TIME_ST, dprint

warnings.simplefilter('ignore')

SAVE_PATH = f'../data_prepaired/{DATASET_NAME}'
DATA_PATH = 'data_other/KITTI/training'

ORIG_W, ORIG_H = 1216, 352
STEP_W, STEP_H = 304, 0
CROP_W, CROP_H = 96, 96

try:
    os.mkdir(SAVE_PATH)
    os.mkdir(f'{SAVE_PATH}/img')
    os.mkdir(f'{SAVE_PATH}/gt')
except FileExistsError:
    pass

DATASET_NAME = DATA_PATH.split('/')

for folder in ['img', 'gt']:
    time_start = time.time()
    all_files = os.listdir(f'../{DATA_PATH}/{folder}')
    dprint('processes', f'Starting splitting in /{folder}')
    dprint('processes', f'Total files in folder: {len(all_files)}')

    for ind, file in enumerate(all_files):
        if time.time() - time_start > 2.0:
            tmp = str(round(100 * ind / len(all_files), 2))
            tmp += '0' * (5 - len(tmp))
            dprint('more_proc', f'Splitting {ind} image [{tmp}%]')
            time_start = time.time()

        img = imread(f'../{DATA_PATH}/{folder}/{file}')
        w, h = 0, 0

        if STEP_H == 0:
            h_range = 1
        else:
            h_range = math.ceil(ORIG_H / STEP_H)

        if STEP_W == 0:
            w_range = 1
        else:
            w_range = math.ceil(ORIG_W / STEP_W)

        for h_crp in range(h_range):
            w = 0
            if h + CROP_H > img.shape[0]:
                h = img.shape[0] - CROP_H
            for w_crp in range(w_range):
                if w + CROP_W > img.shape[1]:
                    w = img.shape[1] - CROP_W
                crp_img = img[h:h + CROP_H, w:w + CROP_W]
                # plt.imshow(crp_img); plt.axis('off'); plt.show()
                imsave(f'{SAVE_PATH}/{folder}/{file[:-4]}_{h_crp}{w_crp}.png', crp_img)
                w += STEP_W
            if h == img.shape[0] - CROP_H:
                break
            h += STEP_H

    dprint('processes', 'Splitting done')

dprint('processes', f'Program done, eval time {round(time.time() - TIME_ST, 2)} sec')