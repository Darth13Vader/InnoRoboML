from PIL import Image, ImageDraw
from random import randint
import os

COUNT = 50
FOLDER = 'data'

try:
    os.mkdir(FOLDER)
except FileExistsError:
    pass

try:
    os.mkdir(f'{FOLDER}/origin')
except FileExistsError:
    pass

try:
    os.mkdir(f'{FOLDER}/mask')
except FileExistsError:
    pass


def solve(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1])) ** 0.5


for i in range(1, COUNT + 1):
    img = Image.new("RGB", (32, 32), (50, 50, 50))
    mask = Image.new("RGB", (32, 32), (0, 0, 0))

    src_draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)

    x, y, r = randint(1, 32), randint(1, 32), randint(1, 16)

    tri = [
        (randint(1, 32), randint(1, 32)),
        (randint(1, 32), randint(1, 32)),
        (randint(1, 32), randint(1, 32))
    ]

    square = [(randint(1, 32), randint(1, 32)), (randint(1, 32), randint(1, 32))]

    src_draw.rectangle(square, fill=(255, 0, 0))
    src_draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 255))
    src_draw.polygon(tri, fill=(0, 255, 0))

    mask_draw.rectangle(square, fill=(5, 5, 5))
    mask_draw.ellipse((x - r, y - r, x + r, y + r), fill=(10, 10, 10))
    mask_draw.polygon(tri, fill=(15, 15, 15))

    img.save(f'{FOLDER}/origin/img{i}.png')
    mask.save(f'{FOLDER}/mask/img{i}.png')
