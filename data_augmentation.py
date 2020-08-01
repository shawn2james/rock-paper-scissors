from PIL import Image
from random import randint
import os
from tqdm import tqdm

def flip_and_save(image_path):
    """
    flip the image by the path and save the flipped image with suffix 'flip'
    :param path: the path of specific image
    :return: the path of saved image
    """
    [image_name, image_ext] = os.path.splitext(os.path.basename(image_path))
    image_dir = os.path.dirname(image_path)
    saved_image_path = os.path.join(
        image_dir, image_name + '_flip' + image_ext)

    angle = randint(-15, 15)
    flipped_image = Image.open(image_path)
    flipped_image = flipped_image.rotate(angle, expand=True)
    flipped_image.save(saved_image_path)
    return saved_image_path


for folder in os.listdir('my-images/train'):
    print(folder)
    folder_path = f'my-images/train/{folder}'
    images = os.listdir(folder_path)
    for img in tqdm(images):
        flip_and_save(f'{folder_path}/{img}')

