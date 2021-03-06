import math
import os
import threading
import time
import warnings
from skimage.io import imread, imsave

# ==================== Govnokod,sorre ==================== #
TIME_ST = time.time()
DEBUG_LEVELS = {'print all': True, 'variables': True, 'processes': True, 'more_proc': True}
load_prepaired = False
cfg_kamaz_dat = {'model_name':            'kamaz',
                 'dataset_path':          'data_prepaired/kamaz' if load_prepaired else 'data_kamaz',
                 'dataset_images_folder': 'img',
                 'dataset_labels_folder': 'masks_machine',
                 'dataset_size':          (512, 512, 3) if load_prepaired else (1280, 1024, 3),
                 'random_seed':           42}


def dprint(level: str, *values, sep=' ', end='\n'):
    if DEBUG_LEVELS['print all'] or DEBUG_LEVELS[level]:
        tm = str(round(time.time() - TIME_ST, 2))
        tm += '0' * (5 - len(tm))
        tm = '0' * (6 - len(tm)) + tm
        print(f'[{tm}][{level}] - ', end='')
        print(*values, sep=sep, end=end)


# ==================== Govnokod ended ==================== #

warnings.simplefilter('ignore')

SAVE_PATH = f'../data_prepaired/kamaz'
DATA_PATH = 'data_kamaz'

ORIG_W, ORIG_H = 1280, 1024
STEP_W, STEP_H = 430, 350
CROP_W, CROP_H = 512, 512
THREADS = 500

folder_img = cfg_kamaz_dat["dataset_images_folder"]
folder_lbl = cfg_kamaz_dat["dataset_labels_folder"]

try:
    os.mkdir(SAVE_PATH)
    os.mkdir(f'{SAVE_PATH}/{folder_img}')
    os.mkdir(f'{SAVE_PATH}/{folder_lbl}')
except FileExistsError:
    pass

DATASET_NAME = DATA_PATH.split('/')


def split_image(file):
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


for folder in [folder_img, folder_lbl]:
    time_start = time.time()
    all_files = os.listdir(f'../{DATA_PATH}/{folder}')
    dprint('processes', f'Starting splitting in /{folder}')
    dprint('processes', f'Total files in folder: {len(all_files)}')

    for ind, file in enumerate(all_files):
        while threading.active_count() > THREADS: pass

        if time.time() - time_start > 2.0:
            tmp = str(round(100 * ind / len(all_files), 2))
            tmp += '0' * (5 - len(tmp))
            dprint('more_proc', f'Splitting {ind} image [{tmp}%]')
            time_start = time.time()

        threading.Thread(target=split_image, args=[file]).start()

    dprint('processes', 'Waiting threadings...')
    while threading.active_count() > 1: pass

    dprint('processes', 'Splitting done')

dprint('processes', f'Program done, eval time {round(time.time() - TIME_ST, 2)} sec')
