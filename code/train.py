# -*- coding: utf-8 -*-

import data_generator_v4 as data_generator
import keys
import os, glob 
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from imp import reload
import densenet
import model as models

img_h = 32
img_w = 280


if __name__ == '__main__':  
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from imp import reload
    reload(keys)
    reload(data_generator)
    reload(models)
    
    K.clear_session()
    
    
    
    characters = keys.alphabet[:]
    characters = characters[1:] + u'卍'
    
    tti = data_generator.TexttoImg(characters)
    print('The number of characters', tti.nclass)
    
    basemodel, model = models.get_model(img_h, tti.nclass)
    
    gen_train = tti.generator_of_ctc(batch_size=128,
                                     input_shape=(32,480,1),
                                     shuffle_text=True,
                                     mode='train',
                                     path='../corpus/address_mini.txt') 

    gen_valid = tti.generator_of_ctc(batch_size=64,
                                     input_shape=(32,480,1),
                                     shuffle_text=False,
                                     mode='test',
                                     path='../corpus/address_mini.txt')
    
    
    model_path = '../log/ocr-6308-0.10.h5'
    
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
