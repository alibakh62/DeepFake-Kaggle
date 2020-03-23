import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import cv2 as cv

# configs
MODELS_FOLDER = '/content/drive/My Drive/DeepFake/models'
BASE_FOLDER = '/content/drive/My Drive/DeepFake'
TRAIN_SAMPLE_VIDEOS = 'train_sample_videos'
TRAIN_INPUT = 'input'
TEST_VIDEOS = 'test_videos'
TRAIN_FRAMES_FOLDER = 'train_frames'
VALID_FRAMES_FOLDER = 'valid_frames'
TEST_FRAMES_FOLDER = 'test_frames'
DATA_FOLDER = 'data'
TRAIN_FOLDER = 'train'
VALID_FOLDER = 'valid'
TEST_FOLDER = 'test'

def get_frames(cap, name, dirname, resize=None, test=False):
    frameRate = cap.get(5) #frame rate
    if not test:
        label = meta_train_df.loc[meta_train_df.index == name, "label"].values[0]
    count = 0
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames in a different folder
            if resize is not None:
                # image = np.array(tf.image.resize(frame, resize))
                image = cv.resize(frame, resize)
            else:
                image = frame
            count += 1
            fn = os.path.splitext(name)[0] + f"_frame{count}.jpg"
            if test:
                file_path = os.path.join(BASE_FOLDER, dirname, fn)
            else:
                file_path = os.path.join(BASE_FOLDER, dirname, label, fn)
            cv.imwrite(file_path, image)
    cap.release()

def walkdir(dirpath):
    for root, dirs, files in os.walk(dirpath):
        for name in files:
            if name != 'metadata.json':
                yield os.path.abspath(os.path.join(dirpath, name)), name

input_folder_name = 'dfdc_train_part_1' #@param {type:"string"}

def get_meta_from_json(path):
    df = pd.read_json(os.path.join(BASE_FOLDER, path, json_file))
    df = df.T
    return df


train_list = list(os.listdir(os.path.join(BASE_FOLDER, TRAIN_INPUT, input_folder_name)))
json_file = [file for file in train_list if  file.endswith('json')][0]
meta_train_df = get_meta_from_json(os.path.join(TRAIN_INPUT, input_folder_name))

train_val_split = 0.8
filescount = meta_train_df.index.nunique()
traincount = int(filescount * train_val_split)

cnt = 1
for path, name in tqdm_notebook(walkdir(os.path.join(BASE_FOLDER, TRAIN_INPUT, input_folder_name)), total=filescount):
    cap = cv.VideoCapture(path)
    if cnt <= traincount:
        dirname = os.path.join(DATA_FOLDER, TRAIN_FOLDER)
    else:
        dirname = os.path.join(DATA_FOLDER, VALID_FOLDER)
    get_frames(cap, name, dirname)
    cnt += 1