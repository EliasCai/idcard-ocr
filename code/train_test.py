# -*- coding: utf-8 -*-

import data_generator_v2 as data_generator
import keys
import os, glob, sys
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import numpy as np
from PIL import Image
from imp import reload
import densenet
import model as models

img_h = 32
img_w = 280


if __name__ == '__main__':  
    
    from imp import reload
    reload(keys)
    reload(data_generator)
    reload(models)
    
    K.clear_session()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    characters = keys.alphabet[:]
    characters = characters[1:] + u'卍'
    
    tti = data_generator.TexttoImg(characters)
    print('The number of characters', tti.nclass)
    
    basemodel, model = models.get_model(img_h, tti.nclass)
    
    gen_train = tti.generator_of_ctc(batch_size=128,
                                     input_shape=(32,280,1),
                                     shuffle_text=True,
                                     mode='train',
                                     path='../corpus/address_mini.txt') 

    gen_valid = tti.generator_of_ctc(batch_size=64,
                                     input_shape=(32,280,1),
                                     shuffle_text=False,
                                     mode='test',
                                     path='../corpus/address_mini.txt')
    
    
    
    inputs, outputs = next(gen_train)
    for idx, img_np in enumerate(inputs['the_input']):
        img = np.squeeze(img_np)
        img = (img + 0.5)*255
        img = Image.fromarray(img.astype(np.uint8))
        img.save('../output/%d.jpg' % idx)
    sys.exit(0)
    model_path = '../log/ocr-6308-0.07.h5'
    
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print('Loading weight from', model_path)
    
    checkpoint = ModelCheckpoint(
                    filepath='../log/ocr-%d-{val_loss:.2f}.h5' % tti.nclass, 
                    monitor='val_loss', 
                    save_best_only=True, 
                    save_weights_only=True,verbose=1,period=5)
    
    model.fit_generator(gen_train,
                        steps_per_epoch=100,    	
                        epochs=1000,
                        validation_data=gen_valid,
                        validation_steps=10,
                        callbacks=[checkpoint],
                        max_q_size=64,
                        workers=2)
