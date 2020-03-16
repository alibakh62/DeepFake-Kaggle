import os
import sys
import logging

sys.path.insert(0, '.')
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2 as cv

import dlib
from tensorflow.keras.preprocessing import image
import multiprocessing
from joblib import Parallel, delayed
from glob import glob

logging.basicConfig(format='%(name)s - %(levelname)s - %(asctime)s - %(filename)s - %(lineno)d - %(message)s', level=logging.DEBUG)

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

images = glob(os.path.join(BASE_FOLDER, DATA_FOLDER, VALID_FOLDER, 'REAL', '*.jpg'))
face_dir = os.path.join(BASE_FOLDER, DATA_FOLDER, VALID_FACE, 'REAL')
logging.info(f"Extracting faces for: {face_dir}")
for idx, i in enumerate(images):
    framename = i.split('/')[-1]
    name = framename.split("_")[0]
    frame_num = os.path.splitext(framename.split("_")[1])[0]
    logging.info(f"reading image: {name}")
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
                    cv.imwrite(file_path, roi_)
    except Exception as e:
        logging.ERROR(e)
        pass
                
