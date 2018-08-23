import os
from random import randint
from skimage.util import random_noise
from skimage.io import imsave, imread
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")


class DataGenerator:
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self._defect_methods = [self._rotate_horizontal, self._rotate_vertical, self._blur, random_noise]

    @staticmethod
    def _rotate_horizontal(img):
        return img[:, :: -1]

    @staticmethod
    def _rotate_vertical(img):
        return img[:: -1, :]

    @staticmethod
    def _blur(img):
        return ndimage.uniform_filter(img, size=(11, 11, 1))

    def data_augmentation(self, random=False):
        try:
            os.mkdir("new_data/")
            os.mkdir("new_data/img")
            os.mkdir("new_data/masks")
        except FileExistsError:
            pass

        imgs_names = sorted(filter(lambda x: x[-4:] == ".jpg", os.listdir(self.images_path)))
        masks_names = sorted(filter(lambda x: x[-4:] == ".png", os.listdir(self.masks_path)))
        name = 0
        for img_name, mask_name in zip(imgs_names, masks_names):
            if random:
                defects = []
                for i in range(randint(0, len(self._defect_methods) - 1)):
                    while True:
                        defect = self._defect_methods[randint(0, len(self._defect_methods) - 1)]
                        if defect not in defects:
                            defects.append(defect)
                            break
            else:
                defects = self._defect_methods
            image = imread(self.images_path + img_name)
            mask = imread(self.masks_path + mask_name)
            for i in range(len(defects)):
                imsave("new_data/img/{}.jpg".format(name), defects[i](image))
                if defects[i] != random_noise:
                    imsave("new_data/masks/{}.jpg".format(name), defects[i](mask))
                else:
                    imsave("new_data/masks/{}.jpg".format(name), mask)

                name += 1
            imsave("new_data/img/{}.jpg".format(name), image)
            imsave("new_data/masks/{}.png".format(name), mask)
            name += 1
            if name % 10 == 0:
                print(name)


if __name__ == '__main__':
    dg = DataGenerator("../img/", "../masks_machine/")
    dg.data_augmentation(False)
