# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os, sys, cv2, uuid, base64, glob
import data_generator 
import keys
from keras import backend as K
from keras.models import Model
import densenet
import model as models
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import json


# 结果图片存储路径
save_dir = '/data1/ID_CARD/'
img_h = 32
img_w = 280

path_label = 'id_label_map.pbtxt'
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(path_label)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_detect_model(path_ckpt=None):
        
    tf.reset_default_graph()
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    detection_graph.as_default()
    sess = tf.Session(graph=detection_graph) 
    
    return sess, detection_graph 

    
def test_case():

    sess_detect, graph = load_model3()
    test_path = '/data/IDCARD20180815/身份证第一批（0731）/4/JPEGImages/'
    img_list =  glob.glob(os.path.join(test_path, '*.jpg'))
    if not os.path.exists('test_case'):
        os.mkdir('test_case')
    
    idx = 0
    np.random.shuffle(img_list)
    
    for img in img_list[:5]:
        
        img_np = cv2.imread(img)
        label_box = test_model3(sess_detect, graph, img_np)
        print(img)
        for label, box in label_box.items():
            box.save(os.path.join('test_case', str(label) + str(idx) + '.jpg'))
            idx += 1
    

def detect_text(sess, detection_graph, image_np):
    
    image = Image.fromarray(image_np)
    image_np = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np})
    
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)
    
    label_box = {}
    
    for idx, v in category_index.items():
        cond = np.column_stack([classes == idx, scores > 0.9]).all(axis=1) # 4表示sim卡
        if boxes[cond].shape[0] == 0: # 如果检测不到制定的物体
            continue
        else:
            points_list = []
            for box in boxes[cond]:
                points = box
                points = points[[1,0,3,2]]
                points = points * [image.size[0],image.size[1],image.size[0],image.size[1]]
                points = points.astype(np.int)
                if v['name'] in ['valid','idnum']:
                    points[2] = points[2] + 10

                label_box[v['name']] = image.crop(points) 
    return label_box

class ID_OCR(object):

    def __init__(self, path_ckpt=None, model_path=None):
        
        if path_ckpt is None:
            path_ckpt = '../detect/frozen_inference_graph.pb' # 模型权重
        
        if model_path is None:
            model_path = '../log/ocr-6308-0.16.h5'
        
        self.sess_detect, self.graph = load_detect_model(path_ckpt)
        
        
        characters = keys.alphabet[:]
        characters = characters[1:] + u'卍'
        self.tti = data_generator.TexttoImg(characters)
        # print(tti.nclass)
        basemodel, model = models.get_model(img_h, self.tti.nclass)
        
        
    
        if os.path.exists(model_path):
            model.load_weights(model_path)
            print('Loading weight from', model_path)
        
        y_pred = model.get_layer('out').output
        input_data = model.get_layer('the_input').input
        
        self.ocr_func = K.function([input_data] + [K.learning_phase()], [y_pred])
                
    def predict(self, img_np):

        label_box = detect_text(self.sess_detect, self.graph, img_np)
        
        #  结果字符串 初始化
        data = {}
        data['type'] = ''
        data['address'] = ''
        data['birthday'] = ''
        data['gender'] = ''
        data['id_card_number'] = ''
        data['name'] = ''
        data['race'] = ''
        data['side'] = ''
        data['issued_by'] = ''
        data['valid_date'] = ''
        data['else'] = ''
        
        target_box = ['name', 'gender']
        target_num = sum([1 for k,v in label_box.items() if k in target_box]) # 
        # print(target_num)
        
        for label, box in label_box.items():
            
            box = box.resize((280,32))
            box = box.convert('L')
            box.save('../output/%s.jpg'%label)
            
            # img = Image.open(test_img_path)
            img_np = data_generator.prepare_img(box)
            # img_np = np.expand_dims(img_np,0)
        
            num_decode = models.decode_batch(self.ocr_func, [img_np])
            text_decode = self.tti.num_to_text(num_decode[0])
            logit = ''.join(text_decode)
            # logit = model.predict(box, basemodel)
            logit = logit.replace('.', '')
            # print(label, logit)
            if label in ['name', 'gender']:
                data[label] = logit
            if label == 'nation':
                data['race'] = logit
            if label == 'birthdate':
                data['birthday'] = logit
            if label == 'idnum':
                data['id_card_number'] = logit
            if label in ['address1', 'address2', 'address3']:
                data['address'] += logit
            
            if label == 'issued':
                data['issued_by'] = logit
                data['side'] = 'back'
            
            if label == 'valid':
                data['valid_date'] = logit
                data['side'] = 'back'
                
        if data['side'] != 'back':
            data['side'] = 'front'
        data = json.dumps(data, ensure_ascii=False)
        return data
        
if __name__ == '__main__':
    
    from imp import reload
    reload(keys)
    reload(data_generator)
    reload(models)
    
    K.clear_session()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    front_img_paths = glob.glob('../eval_data/ID_COVER/*.jpg')
    back_img_paths = glob.glob('../eval_data/BACK_RAW/*.jpg')
    
    img_path = front_img_paths[5]
    img_np = cv2.imread(img_path)
    
    id_ocr = ID_OCR(model_path='../log/ocr-6308-0.16.h5')
    
    decode_data = id_ocr.predict(img_np)
    print(decode_data)
    