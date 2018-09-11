import data_generator_v2 as data_generator
import keys
import os, glob, itertools, random

from keras import backend as K
from keras.models import Model
import densenet
import model as models
from PIL import Image
import numpy as np

img_h = 32
img_w = 280


if __name__ == '__main__':  
    
    from imp import reload
    reload(keys)
    reload(data_generator)
    reload(models)
    
    K.clear_session()
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    characters = keys.alphabet[:]
    characters = characters[1:] + u'卍'
    
    tti = data_generator.TexttoImg(characters)
    print(tti.nclass)
    
    basemodel, model = models.get_model(img_h, tti.nclass)
    
    model_path = '../log/ocr-6308-0.07.h5'
    
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print('Loading weight from', model_path)
        
    y_pred = model.get_layer('out').output
    input_data = model.get_layer('the_input').input
    test_img_paths = glob.glob(os.path.join('../output','*.jpg'))
    
    ocr_func = K.function([input_data] + [K.learning_phase()], [y_pred])
    

    for test_img_path in random.sample(test_img_paths,5):
    
        img = Image.open(test_img_path)
        img_np = data_generator.prepare_img(img)
        # img_np = np.expand_dims(img_np,0)
        
        num_decode = models.decode_batch(ocr_func, [img_np])
        
        text_decode = tti.num_to_text(num_decode[0])
        
        text_decode = ''.join(text_decode)
        img_name = os.path.basename(test_img_path).replace('.jpg','')
        
        print(text_decode, img_name, text_decode == img_name)
    