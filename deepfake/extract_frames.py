"""
    - run this in the command line using nohup as below:

    `nohup python -u extract_frames.py > train_frames.log &`

    - you can check if the code is running by executing this:

    `ps ax | grep extract_frames.py`

    NOTE: make sure that you've activated the virtual environment
    NOTE: for extracting frames from test videos, add the '-t' argument:
          `nohup python -u extract_frames.py -t > train_frames.log &`

    good source for nohup: https://janakiev.com/blog/python-background/
"""

import os
import sys
import logging

sys.path.insert(0, '.')
sys.path.insert(0, '..')

import math
import numpy as np
import pandas as pd
import cv2 as cv
from glob import glob
import multiprocessing
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='DeepFake DataPrep Parser')
parser.add_argument('-t', action="store_true", default=False, help="Is it preprocessing for test data")
args = parser.parse_args()
global test, metadata
test = args.t

logging.basicConfig(format='%(name)s - %(levelname)s - %(asctime)s - %(filename)s - %(lineno)d - %(message)s', level=logging.DEBUG)

# configs
BASE_FOLDER = '..'
DATA_FOLDER = 'data'
TRAIN_FRAMES = 'train'
VALID_FRAMES = 'valid'
TEST_FRAMES = 'test'
TRAIN_VIDEOS = 'train_videos'
CHUNK_NAME = 'train_sample_videos'
TEST_VIDEOS = 'test_videos'
METADATA = 'metadata'

# ========== creating directories
logging.info("Creating directories (if needed)")
data_dir = os.path.join(BASE_FOLDER, DATA_FOLDER)

# train frames folder
train_dir = os.path.join(data_dir, TRAIN_FRAMES)
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

if not os.path.isdir(os.path.join(train_dir, 'REAL')):
    os.mkdir(os.path.join(train_dir, 'REAL'))

if not os.path.isdir(os.path.join(train_dir, 'FAKE')):
    os.mkdir(os.path.join(train_dir, 'FAKE'))

# valid frames folder
valid_dir = os.path.join(data_dir, VALID_FRAMES)
if not os.path.isdir(valid_dir):
    os.mkdir(valid_dir)

if not os.path.isdir(os.path.join(valid_dir, 'REAL')):
    os.mkdir(os.path.join(valid_dir, 'REAL'))

if not os.path.isdir(os.path.join(valid_dir, 'FAKE')):
    os.mkdir(os.path.join(valid_dir, 'FAKE'))

# test frames folder
test_dir = os.path.join(data_dir, TEST_FRAMES)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)


# ========== getting a count of frames in each folder
logging.info(f"Total train videos: {len(os.listdir(os.path.join(BASE_FOLDER, DATA_FOLDER, TRAIN_VIDEOS, CHUNK_NAME)))}")
logging.info(f"Total test videos: {len(os.listdir(os.path.join(BASE_FOLDER, DATA_FOLDER, TEST_VIDEOS)))}")


# ========== functions
def get_metadata(jsonpath):
    metadf = pd.DataFrame()
    json_pattern = os.path.join(jsonpath, '*.json')
    flist = glob(json_pattern)
    for f in flist:
        df = pd.read_json(f)
        df = df.T
        metadf = metadf.append(df, ignore_index = False)
    return metadf

def get_frames(cap, name, dirname, metadata, test=False):
    frameRate = cap.get(5) #to get the frame rate
    if not test:
        label = metadata.loc[metadata.index == name, "label"].values[0]
    count = 0
    logging.info(f"capturing frames of video file: {name}")
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            logging.info("frames couldn't be captured!")
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames in a different folder
            image = frame
            count += 1
            logging.info(f"frame number: {count}")
            fn = os.path.splitext(name)[0] + f"_frame{count}.jpg"
            if test:
                file_path = os.path.join(dirname, fn)
            else:
                file_path = os.path.join(dirname, label, fn)
            cv.imwrite(file_path, image)
    cap.release()
    logging.info("="*50)

def get_video_path(datadir, test=False):
    if test:
        video_path = os.path.join(datadir, TEST_VIDEOS)
    else:
        video_path = os.path.join(datadir, TRAIN_VIDEOS, CHUNK_NAME)
    return video_path

def main(cnt, video_path):
    name = video_path.split('/')[-1]
    cap = cv.VideoCapture(video_path)
    cnt += 1
    if not test:
        if cnt <= traincount:
            dirname = os.path.join(BASE_FOLDER, DATA_FOLDER, TRAIN_FRAMES)
        else:
            dirname = os.path.join(BASE_FOLDER, DATA_FOLDER, VALID_FRAMES)
    else:
        dirname = os.path.join(BASE_FOLDER, DATA_FOLDER, TEST_FRAMES)
    get_frames(cap, name, dirname, metadata, test)


if __name__ == "__main__":
    datadir = os.path.join(BASE_FOLDER, DATA_FOLDER)
    video_path = get_video_path(datadir)
    videos_list = glob(os.path.join(video_path, '*.mp4'))
    train_val_split = 0.8
    filescount = len(videos_list)
    traincount = int(filescount * train_val_split)
    metadata = get_metadata(os.path.join(datadir, METADATA))
    pool = multiprocessing.Pool()
    pool.starmap(main, enumerate(videos_list))  #starmap for multiple args
    pool.close()
    logging.info(f"Total REAL train: {len(os.listdir(os.path.join(BASE_FOLDER, DATA_FOLDER, TRAIN_FRAMES, 'REAL')))}")
    logging.info(f"Total FAKE train: {len(os.listdir(os.path.join(BASE_FOLDER, DATA_FOLDER, TRAIN_FRAMES, 'FAKE')))}")
    logging.info(f"Total REAL valid: {len(os.listdir(os.path.join(BASE_FOLDER, DATA_FOLDER, VALID_FRAMES, 'REAL')))}")
    logging.info(f"Total FAKE valid: {len(os.listdir(os.path.join(BASE_FOLDER, DATA_FOLDER, VALID_FRAMES, 'FAKE')))}")
    logging.info(f"Total frames test: {len(os.listdir(os.path.join(BASE_FOLDER, DATA_FOLDER, TEST_FRAMES)))}")