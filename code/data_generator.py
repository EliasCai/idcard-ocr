# -*- coding: utf-8 -*-



from PIL import Image,ImageFont,ImageDraw
import numpy as np
import random
import os, sys, codecs
import keras.backend as K
import keys
from keras.preprocessing.sequence import pad_sequences


def strQ2B(ustring): # 全角符号转半角
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)



class TexttoImg():
    
    def __init__(self, char_set):
        
        self.bg = []
        img_bg = Image.open('../background/front.jpg')
        self.bg.append(img_bg)
        img_bg = Image.open('../background/back.jpg')
        self.bg.append(img_bg)
        img_bg = Image.new("RGB", (800, 800),(255,255,255)) 
        self.bg.append(img_bg)
        
        self.font = []
        self.font_size = 15
        font = ImageFont.truetype("../font/STFANGSO.TTF", self.font_size)  
        self.font.append(font)
        
        # 转成字母与数字的键值对，注意下标从1开始
        self.char_set = char_set
        self.letter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        self.char_to_num = dict((char, idx+1) \
                                for idx, char in enumerate(char_set))
        
        self.num_to_char = dict((idx, char) \
                                for char, idx in self.char_to_num.items())
        self.num_to_char[0] = ''
        self.to_num = lambda x: self.char_to_num[x]
        self.to_char = lambda x: self.num_to_char[x]
        
        self.nclass = max(self.num_to_char.keys()) + 2
        
    def draw_text(self, txt, mode):  
        
    #    img = Image.new("RGB", (280, 32),(255,255,255))  
        if mode == 'train':
            bg = random.choice(self.bg) # 随机挑选背景
            font = random.choice(self.font) # 随机挑选字体
            img = bg.crop((10,10,len(txt)*self.font_size + 10,30))
            draw = ImageDraw.Draw(img)  
            draw.text((0, 0), txt, font=font, fill=(0,0,0))  
            degree = np.random.randint(-3,3)
            img = img.rotate(degree, expand=False)
            img = img.resize((280,32))
            img = img.convert('L')
            img_np = np.array(img).astype(np.float32)
            img_np = np.expand_dims(img_np,2)
            noise = np.random.normal(0,0.5,img_np.shape)
            img_np = img_np + noise
        else:
            bg = self.bg[2] # 白色背景
            font = random.choice(self.font) # 随机挑选字体
            img = bg.crop((10,10,len(txt)*self.font_size + 10,30))
            draw = ImageDraw.Draw(img)  
            draw.text((0, 0), txt, font=font, fill=(0,0,0))  
            img = img.resize((280,32))
            img = img.convert('L')
#            img.save("../output/text.jpg")
            img_np = np.array(img).astype(np.float32)
            img_np = np.expand_dims(img_np,2)
        img_np = img_np / 255. - 0.5
        return img_np
    
    def text_to_num(self, txt):
        
        num_list = list(map(self.to_num, txt))
        return num_list
  
    def num_to_text(self, num_list):
        
        text = list(map(self.to_char, num_list))
        return text
    
    def generator_of_corpus(self, batch_size, path):
        
        with codecs.open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [strQ2B(line.encode('utf-8').decode('utf-8-sig').strip()) \
                 for line in lines] # 去掉\ufeff
        lines = [line.replace(' ','') for line in lines if len(line) > 0] # 去掉空行
        while True:
            
            batch_lines = random.sample(lines, batch_size)
#            batch_lines = lines
            yield batch_lines
            
    def generator_of_xy(self, batch_size, shuffle_text, mode, path):
        gen_corpus = self.generator_of_corpus(batch_size, path)
        while True:
            x, y, texts = [], [], []
            lines = next(gen_corpus)
            for line in lines:
                if shuffle_text and random.choice([True, False]):
                    random_str = random.sample(self.char_set,62)
                    random_str = random.sample(
                        self.letter + ''.join(random_str),22)
                    text = ''.join(random_str)
                else:
                    text = line
                if shuffle_text:
                    text = ''.join(
                            random.sample(text, 
                            random.randint(len(text) // 2,len(text))))
                text = text[:20]
                texts.append(text)
                img_np = self.draw_text(text,mode=mode)
                num_of_text = self.text_to_num(text)
                x.append(img_np)
                y.append(num_of_text)
            x = np.array(x)
            y = pad_sequences(y,maxlen=24,padding='post', truncating='post')
            yield x, y, texts
            
    def generator_of_ctc(self, 
                         batch_size=32,
                         input_shape=(32,280,1),
                         shuffle_text=True,
                         mode='test',
                         path='../corpus/address_mini.txt'):
        gen_xy = self.generator_of_xy(batch_size=batch_size,
                                      shuffle_text=shuffle_text,
                                      mode=mode,
                                      path=path)
        while True:
            x, y, texts = next(gen_xy)
#            input_length = np.zeros([batch_size, 1])
#            label_length = np.zeros([batch_size, 1])
            input_length = \
                np.array([input_shape[1] // 8] * batch_size).reshape(-1,1)
            label_length = np.array([len(t) for t in texts]).reshape(-1,1)
            inputs = {'the_input': x,
                'the_labels': y,
                'input_length': input_length,
                'label_length': label_length,
                'texts': texts
                }
            
            outputs = {'ctc': np.zeros([batch_size])}
            yield (inputs, outputs)
            
if __name__ == '__main__':  
    from imp import *
    reload(keys)
    K.clear_session()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    characters = keys.alphabet[:]
    characters = characters[1:] + u'卍'
    with codecs.open('../corpus/address_mini.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
    lines = [strQ2B(line.encode('utf-8').decode('utf-8-sig').strip()) \
             for line in lines] # 去掉\ufeff
    lines = [line for line in lines if len(line) > 0] # 去掉空行

    tti = TexttoImg(characters)
    print(tti.nclass)
    gen_train = tti.generator_of_ctc(batch_size=32,
                                     input_shape=(32,280,1),
                                     shuffle_text=True,
                                     mode='test',
                                     path='../corpus/address_mini.txt')
    for i in range(10):
        inputs, outputs = next(gen_train)
    
    for k,v in inputs.items():
        print(k, len(v))
    
    # for i in range(10000):
        # random_str = random.sample(tti.char_set,62)
        # random_str = random.sample(
            # tti.letter + ''.join(random_str),24)
        # text = ''.join(random_str)
    
    # print(i, text)
    # print(tti.text_to_num(text))
