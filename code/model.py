# -*- coding: utf-8 -*-

# import data_generator 
import keys
import os, glob, itertools
import tensorflow as tf
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten
from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
import numpy as np
from imp import reload
import densenet


img_h = 32
img_w = 280

def decode_batch(test_func, batch):
    out = test_func([batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, :], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        ret.append(out_best)
    return ret

def ctc_lambda_func(args):
    
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(img_h, nclass):
    
    input_layer = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_cnn(input_layer, nclass)

    basemodel = Model(inputs=input_layer, outputs=y_pred)
#    basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, 
                      output_shape=(1,), 
                      name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_layer, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])

    return basemodel, model

if __name__ == '__main__':  
    
    from imp import *
    reload(keys)
    reload(data_generator)
    
    K.clear_session()
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    characters = keys.alphabet[:]
    characters = characters[1:] + u'卍'
    
    tti = data_generator.TexttoImg(characters)
    print(tti.nclass)
    
    basemodel, model = get_model(img_h, tti.nclass)
    
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
    
    
    model_path = '../log/ocr-6308-0.04.h5'
    
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print('Loading weight from', model_path)
    
    checkpoint = ModelCheckpoint(
                    filepath='../log/ocr-%d-{val_loss:.2f}.h5' % tti.nclass, 
                    monitor='val_loss', 
                    save_best_only=True, 
                    save_weights_only=True,verbose=1,period=5)
    

    # for i in range(1000):
        # inputs, outputs = next(gen_train)
        
        # model.train_on_batch(inputs, outputs)

        # if i % 100 == 0:
            # print(i)
    
    model.fit_generator(gen_train,
                        steps_per_epoch=200,    	
                        epochs=1000,
                        validation_data=gen_valid,
                        validation_steps=10,
                        callbacks=[checkpoint],
                        max_q_size=64,
                        workers=2)
