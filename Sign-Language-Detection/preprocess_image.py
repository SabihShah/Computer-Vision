import pandas as pd
import os
import cv2
import random
import numpy as np
import argparse
from imutils import paths
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-images', default=1000, type=int,
    help='number of images to preprocess for each category')
args = vars(parser.parse_args())

print(f"Preprocessing {args['num_images']} from each category...")

# get all the image paths
image_paths = list(paths.list_images('../input/asl_alphabet_train/asl_alphabet_train'))
dir_paths = os.listdir('../input/asl_alphabet_train/asl_alphabet_train')
dir_paths.sort()

root_path = '../input/asl_alphabet_train/asl_alphabet_train'

# get images from each category
for idx, dir_path in tqdm(enumerate(dir_paths), total=len(dir_paths)):
    all_images = os.listdir(f"{root_path}/{dir_path}")
    num_images = min(args['num_images'], len(all_images)) # number of images to preprocess for each category
    os.makedirs(f"../input/preprocessed_image/{dir_path}", exist_ok=True)
    for i in range(num_images):
        rand_id = (random.randint(0, len(all_images)-1)) # generate a random id between 0 and the number of images in the directory
        image = cv2.imread(f"{root_path}/{dir_path}/{all_images[rand_id]}")
        image = cv2.resize(image, (224, 224))

        cv2.imwrite(f"../input/preprocessed_image/{dir_path}/{dir_path}{i}.jpg", image)

print('DONE')
