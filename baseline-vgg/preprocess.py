from hparams import hparams
import pandas as pd
import cv2
from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from skimage import io #scikit-image
from tqdm import tqdm
import os

cant=0

def enhance(image_path, target_path):
    # read the image
    try:
        # read the image
        img = io.imread(image_path)
        # use Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        img_eq = clahe.apply(img)
        # save
        cv2.imwrite(target_path, img_eq)
    except:
        try:
            img = io.imread(image_path)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            #convert to grayscale
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # use Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            img_eq = clahe.apply(gray_image)
            # save
            cv2.imwrite(target_path, img_eq)
        except:
            # cant+=0
            os.system('cp '+image_path+' '+target_path)
            print('cant process.')

# df = pd.read_csv(hparams.train_csv)
# for img_path in tqdm(list(df.iloc[:,0])):
#     enhance(hparams.train_dir+img_path, hparams.pr_train_dir+img_path)
# print('train images saved.')

cant=0
df = pd.read_csv(hparams.valid_csv)
for img_path in tqdm(list(df.iloc[:,0])):
    enhance(hparams.valid_dir+img_path, hparams.pr_valid_dir+img_path)
print('valid images saved.')
#
# cant=0
# df = pd.read_csv(hparams.test_csv)
# for img_path in tqdm(list(df.iloc[:,0])):
#     enhance(hparams.test_dir+img_path, hparams.pr_test_dir+img_path)
# print('test images saved.')
