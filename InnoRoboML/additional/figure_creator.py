import os
from PIL import Image, ImageDraw
from random import randint
from FiguresSegmentation import DATA_PATH, MASK_COLORS, img_w, img_h

COUNT = 5000
FOLDER = '../' + DATA_PATH

try:
    os.mkdir(FOLDER)
except FileExistsError:
    pass

try:
    os.mkdir(f'{FOLDER}/images')
except FileExistsError:
    pass

try:
    os.mkdir(f'{FOLDER}/gt')
except FileExistsError:
    pass


def solve(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1])) ** 0.5


size = img_w
minsize = size // 4
for i in range(1, COUNT + 1):
    img = Image.new("RGB", (size, size), (50, 50, 50))
    mask = Image.new("RGB", (size, size), (0, 0, 0))

    src_draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)

    x, y, r = randint(minsize, size), randint(minsize, size), randint(minsize, size) / 2

    tri = [
        (randint(minsize, size), randint(minsize, size)),
        (randint(minsize, size), randint(minsize, size)),
        (randint(minsize, size), randint(minsize, size))
    ]

    square = [(randint(minsize, size), randint(minsize, size)), (randint(minsize, size), randint(minsize, size))]

    src_draw.rectangle(square, fill=(255, 0, 0))
    src_draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 255))
    src_draw.polygon(tri, fill=(0, 255, 0))

    mask_draw.rectangle(square, fill=tuple([MASK_COLORS[1]] * 3))
    mask_draw.ellipse((x - r, y - r, x + r, y + r), fill=tuple([MASK_COLORS[2]] * 3))
    mask_draw.polygon(tri, fill=tuple([MASK_COLORS[3]] * 3))

    img.save(f'{FOLDER}/images/images{w_crp}.png')
    mask.save(f'{FOLDER}/gt/images{w_crp}.png')