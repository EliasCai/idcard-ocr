import data_generator_v3 as data_generator
import keys
import os, glob, itertools, random, sys

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
    
    model_path = '../log/ocr-6308-0.11.h5'
    
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print('Loading weight from', model_path)
        
    y_pred = model.get_layer('out').output
    input_data = model.get_layer('the_input').input
    test_img_paths = glob.glob(os.path.join('../output','*.jpg'))
    
    ocr_func = K.function([input_data] + [K.learning_phase()], [y_pred])
    
    
    gen_train = tti.generator_of_ctc(batch_size=16,
                                     input_shape=(32,280,1),
                                     shuffle_text=True,
                                     mode='train',
                                     path='../corpus/address_mini.txt') 

    
    
    
    inputs, outputs = next(gen_train)
    for idx, (img_np, text) in enumerate(zip(inputs['the_input'],inputs['texts'])):
    
        num_decode = models.decode_batch(ocr_func, [img_np])
        
        text_decode = tti.num_to_text(num_decode[0])
        
        text_decode = ''.join(text_decode)
        
        print(text_decode, text, text_decode==text)
        
        img = np.squeeze(img_np)
        img = (img + 0.5)*255
        img = Image.fromarray(img.astype(np.uint8))
        img.save('../output/%s.jpg' % text_decode)
    sys.exit(0)
    for test_img_path in random.sample(test_img_paths,5):
    
        img = Image.open(test_img_path)
        img_np = data_generator.prepare_img(img)
        # img_np = np.expand_dims(img_np,0)
        
        num_decode = models.decode_batch(ocr_func, [img_np])
        
        text_decode = tti.num_to_text(num_decode[0])
        
        text_decode = ''.join(text_decode)
        img_name = os.path.basename(test_img_path).replace('.jpg','')
        
        print(text_decode, img_name, text_decode == img_name)
    