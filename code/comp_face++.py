# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os, sys, cv2, uuid, base64, glob
import model

import pandas as pd


import demo_new
# from demo_new import ID_CARD
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


import json
import afterprocessing


def get_json_list(path):
    
    search_path = os.path.join(path, 'ID_JSON', '*.json')
    json_list = glob.glob(search_path)
    
    return json_list

def get_img_list(path):
 
    search_path = os.path.join(path, 'ID_COVER', '*.jpg')
    img_list = glob.glob(search_path)
    
    return img_list

def get_df(json_list, img_list):
    
    get_name = lambda x: os.path.splitext(os.path.basename(x))[0]
    json_dict = dict((get_name(path), path) for path in json_list)
    img_dict = dict((get_name(path), path) for path in img_list)
    inner_name = set(json_dict.keys()) & set(img_dict.keys())
    row = []
    for name in inner_name:
        row.append((name, json_dict[name], img_dict[name]))
    
    df = pd.DataFrame(row, columns=['name','json_path', 'img_path'])
    
    return df

def check_overlap(df): # 检查与训练样本是否重叠
    
    train_path = '/data/IDCARD20180815/Annotations'    
    xml_list = glob.glob(os.path.join(train_path,'*.xml'))
    get_name = lambda x: os.path.splitext(os.path.basename(x))[0]
    xml_dict = dict((get_name(path), path) for path in xml_list)
    name_overlap = set(df['name'].tolist()) & set(xml_dict.keys())
  
    print('The num of overlap ', len(name_overlap))
    
    return xml_dict
    
def get_content(df, model, recog_model):
    
    ocr1_list = []
    ocr2_list = []
    
    for rowid, row  in df.iterrows():
        with open(row['json_path'], 'r') as f:
            try:
                data1 = json.load(f)
            
                data1 = data1['cards'][0]
                data1['img_name'] = df['name']
                ocr1_list.append(data1)
            
            except :
                print(row['json_path']) # 
                continue
        img_np = cv2.imread(row['img_path'])
        data2 = model.predict(img_np, recog_model)
        data2 = json.loads(data2)
        data2['img_name'] = df['name']
        ocr2_list.append(data2)
        # return data1, data2
        # print('reading...',rowid)
    return pd.DataFrame(ocr1_list), pd.DataFrame(ocr2_list)

def comp_content(df_ocr1, df_ocr2):
        
        comp_key = ['name', 'address', 'gender', 'id_card_number', 'race']
        
        for key in comp_key:
            
            accuracy = np.mean(df_ocr1[key] == df_ocr2[key])
            print('the accuracy of', key, '=', round(accuracy,2))

def get_model():

    model = demo_new.ID_CARD()
    recog_model = model.load_model2()
    # img_np = cv2.imread(os.path.join('test', '1341197113120180612153259488862_3.jpg'))
    # output = a.__Cread__(img_np, None, None, basemodel)
    # output = a.predict(img_np, basemodel)
    # print(output)
    return model, recog_model
            
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    base_path = '/data/1808_face-plus-plus'
    
    json_list = get_json_list(base_path) 
    img_list = get_img_list(base_path)
       
    df_face = get_df(json_list, img_list)
    
    model, recog_model = get_model()
    
    df_ocr1, df_ocr2 = get_content(df_face.sample(1000), model, recog_model)
    
    comp_content(df_ocr1, df_ocr2)
