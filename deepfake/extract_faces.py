"""
    - run this in the command line using nohup as below:

    `nohup python -u extract_faces.py -d train -c REAL > train_face_real_logs.log &`

    NOTE: -c can be either REAL or FAKE
    NOTE: -d can be either of train or valid or test

    - you can check if the code is running by executing this:

    `ps ax | grep extract_faces.py`

    NOTE: make sure that you've activated the virtual environment

    good source for nohup: https://janakiev.com/blog/python-background/
"""


import os
import sys
import logging

sys.path.insert(0, '.')
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import cv2 as cv
import dlib
from glob import glob
import multiprocessing
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='DeepFake DataPrep Parser')
parser.add_argument('-c', type=str, help="video class")
parser.add_argument('-d', type=str, help="train/valid/test")
args = parser.parse_args()
global dataset, class_folder
dataset = args.d
class_folder = args.c

logging.basicConfig(format='%(name)s - %(levelname)s - %(asctime)s - %(filename)s - %(lineno)d - %(message)s', level=logging.DEBUG)

# face detector
detector = dlib.get_frontal_face_detector()

BASE_FOLDER = '..'
DATA_FOLDER = 'data'
TRAIN_FOLDER = 'train'
VALID_FOLDER = 'valid'
TEST_FOLDER = 'test'
TRAIN_FACE = 'train_face'
VALID_FACE = 'valid_face'
TEST_FACE = 'test_face'

# ========== creating directories
logging.info("Creating directories (if needed)")
data_dir = os.path.join(BASE_FOLDER, DATA_FOLDER)

# train face folder
train_dir = os.path.join(data_dir, TRAIN_FACE)
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

if not os.path.isdir(os.path.join(train_dir, 'REAL')):
    os.mkdir(os.path.join(train_dir, 'REAL'))

if not os.path.isdir(os.path.join(train_dir, 'FAKE')):
    os.mkdir(os.path.join(train_dir, 'FAKE'))

# valid face folder
valid_dir = os.path.join(data_dir, VALID_FACE)
if not os.path.isdir(valid_dir):
    os.mkdir(valid_dir)

if not os.path.isdir(os.path.join(valid_dir, 'REAL')):
    os.mkdir(os.path.join(valid_dir, 'REAL'))

if not os.path.isdir(os.path.join(valid_dir, 'FAKE')):
    os.mkdir(os.path.join(valid_dir, 'FAKE'))

# test faces folder
test_dir = os.path.join(data_dir, TEST_FACE)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

# ========== getting a count of frames in each folder
logging.info(f"Total REAL train: {len(os.listdir(os.path.join(BASE_FOLDER, DATA_FOLDER, TRAIN_FOLDER, 'REAL')))}")
logging.info(f"Total FAKE train: {len(os.listdir(os.path.join(BASE_FOLDER, DATA_FOLDER, TRAIN_FOLDER, 'FAKE')))}")
logging.info(f"Total REAL valid: {len(os.listdir(os.path.join(BASE_FOLDER, DATA_FOLDER, VALID_FOLDER, 'REAL')))}")
logging.info(f"Total FAKE valid: {len(os.listdir(os.path.join(BASE_FOLDER, DATA_FOLDER, VALID_FOLDER, 'FAKE')))}")


# ========== functions
def rect_to_bb(rect):
    """
       take a bounding predicted by dlib and convert it
       to the format (x, y, w, h) as we would normally do
       with OpenCV
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def main(idx, i):
    cnt_images = 0
    cnt_faces = 0
    cnt_faces_not_detected = 0
    framename = i.split('/')[-1]
    name = framename.split("_")[0]
    frame_num = os.path.splitext(framename.split("_")[1])[0]    
    logging.info(f"reading image: {name}")
    cnt_images += 1
    logging.info(f"count of images read so far: {cnt_images}")
    try:
        img = cv.imread(i)
        faces = detector(img, 1)
        if len(faces) > 0:
            rects, scores, idx = detector.run(img, 0)
            for rect in rects:
                if len(rects) > 0:
                    (x, y, w, h) = rect_to_bb(rect)
                    roi = img[(y - 50):(y + int(1.5*h)), (x - 50):(x + int(1.5*w))]
                    roi_ = cv.resize(roi, (300, 300))
                    fname = f"{name}_{frame_num}.jpg"
                    logging.info(f"Writing file: {fname}")
                    file_path = os.path.join(face_dir, fname)
                    cnt_faces += 1
                    logging.info(f"count of faces read so far: {cnt_faces}")
                    cv.imwrite(file_path, roi_)
        else:
            cnt_faces_not_detected += 1
            logging.info(f"count of face not detected: {cnt_faces_not_detected}")
    except Exception as e:
        logging.error(f"Error occured: {e}")
        pass
    logging.info("="*50)


if __name__ == "__main__":
    
    if dataset == 'test':
        images = glob(os.path.join(BASE_FOLDER, DATA_FOLDER, dataset, '*.jpg'))
    else:
        images = glob(os.path.join(BASE_FOLDER, DATA_FOLDER, dataset, class_folder, '*.jpg'))
    
    if dataset == 'train':
        face_dir = os.path.join(BASE_FOLDER, DATA_FOLDER, TRAIN_FACE, class_folder)
    elif dataset == 'valid':
        face_dir = os.path.join(BASE_FOLDER, DATA_FOLDER, VALID_FACE, class_folder)
    else:
        face_dir = os.path.join(BASE_FOLDER, DATA_FOLDER, TEST_FACE)
    
    logging.info(f"number of frames: {len(images)}")
    logging.info(f"extracting faces into {face_dir}")

    pool = multiprocessing.Pool()
    pool.starmap(main, enumerate(images))
    pool.close()