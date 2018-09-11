# -*- coding: utf-8 -*-
from PIL import Image,ImageFont,ImageDraw, ImageEnhance
import numpy as np
import random
import os, sys, codecs, glob
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

def resize_img(img, resize_method=Image.LANCZOS):
    
    # shape = img.size
    frac = 32 / img.size[1]
    resize_x = int(frac * img.size[0])
    # print(resize_x)
    return img.resize((resize_x, 32), resize_method)
    
def prepare_img(img):

    # img = img.resize((480,32))
    img = resize_img(img)
    
    img = img.convert('L')
    img_np = np.array(img).astype(np.float32)
    img_np = np.expand_dims(img_np,2)
    img_np = img_np / 255. - 0.5
    return img_np
    
class TexttoImg():
    
    def __init__(self, char_set):
        
        self.bg = []
        for bg_path in glob.glob('../background/*.jpg'):
            img_bg = Image.open(bg_path)
            self.bg.append(img_bg.resize((800,800)))
        
        # img_bg = Image.open('../background/back.jpg')
        # shape = img_bg.size
        # self.bg.append(img_bg) # .resize(np.array(shape) * 2))
        
        img_bg = Image.new("RGB", (800, 800),(255,255,255)) 
        self.bg.append(img_bg)
        
        self.resize_method = [Image.NEAREST, Image.BILINEAR, 
                         Image.BICUBIC, Image.LANCZOS]
        
        self.font = []
        self.font_size = 25
        for font_path in glob.glob('../font/*'):
            # font = ImageFont.truetype("../font/STFANGSO.TTF", self.font_size)
            font = ImageFont.truetype(font_path, self.font_size)  
            self.font.append(font)
        
        # 转成字母与数字的键值对，注意下标从1开始
        self.char_set = char_set
        self.letter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + \
                      'abcdefghijklmnopqrstuvwxyz' + \
                      '0123456789' + \
                      '-+.~一' # 容易混淆的标点符号，并随机最多插入4个空格
        self.char_to_num = dict((char, idx+1) \
                                for idx, char in enumerate(char_set))
        
        self.num_to_char = dict((idx, char) \
                                for char, idx in self.char_to_num.items())
        self.num_to_char[0] = ''
        self.to_num = lambda x: self.char_to_num[x]
        
        # self.to_char = lambda x: self.num_to_char[x]
        self.to_char = lambda x: self.num_to_char.get(x,'')
        
        self.nclass = max(self.num_to_char.keys()) + 2
        
    def draw_text(self, txt, mode):  
        
    #    img = Image.new("RGB", (280, 32),(255,255,255))  
        bg = random.choice(self.bg) # 随机挑选背景
        bgsize = bg.size
        font = random.choice(self.font) # 随机挑选字体
        fontsize = font.getsize(txt)
        frac_x = 1.01 + random.random() / 8
        frac_y = 1.01 + random.random() / 3
        cropsize = (max(300, int(fontsize[0] * frac_x)),
                    int(fontsize[1] * frac_y))
        
        random_xmin = random.randint(0,bgsize[0] - cropsize[0])
        random_ymin = random.randint(0,bgsize[1] - cropsize[1])

        img = bg.crop((random_xmin,
                       random_ymin,
                       random_xmin + cropsize[0],
                       random_ymin + cropsize[1]))
        # print(fontsize, cropsize, txt)               
        draw = ImageDraw.Draw(img)
        
        random_xmin = random.randint(0,img.size[0] - fontsize[0])
        random_ymin = random.randint(0,img.size[1] - fontsize[1])
        # print((random_xmin, random_ymin), img.size, fontsize, mode)
        draw.text((random_xmin, random_ymin), 
                  txt, font=font, fill=(0,0,0))  
        
        
        if mode == 'train':
            degree = 0.5 + random.random()
            img = ImageEnhance.Brightness(img).enhance(degree)
            
            degree = 0.5 + random.random()
            img = ImageEnhance.Color(img).enhance(degree)
            
            degree = 0.5 + random.random()
            img = ImageEnhance.Contrast(img).enhance(degree)
            
            degree = 0.5 + random.random()
            img = ImageEnhance.Sharpness(img).enhance(degree)
            
            img = img.convert('L')
            
            degree = random.randint(-5,5) * random.random()
            img = img.rotate(degree, expand=True)
            # img = resize_img(img, np.random.choice(self.resize_method))
            img = img.resize((480,32), np.random.choice(self.resize_method))
            img_np = np.array(img).astype(np.float32)
            img_np = np.expand_dims(img_np,2)
            noise = np.random.normal(0,1,img_np.shape)
            img_np = img_np + noise
            
        else:
            img = img.convert('L')
            # img = resize_img(img, Image.LANCZOS)
            img = img.resize((480,32), Image.LANCZOS)
            img_np = np.array(img).astype(np.float32)
            img_np = np.expand_dims(img_np,2)
        # try :    
            # img = Image.fromarray(np.squeeze(img_np.astype(np.uint8))) 
            # img.save("../output/%s.jpg"%txt)
        # except  :
            # pass
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
        new_lines = []
        
        for line in lines: # 去除不在字典里面的文本行
            try:
                self.text_to_num(line)
            except KeyError:
                continue
            new_lines.append(line)
        print('The number of corpus is', len(new_lines))
        while True:
            batch_lines = random.sample(new_lines, batch_size)
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
                            random.randint(1,len(text))))
                text = text[:20]
                img_np = self.draw_text(text,mode=mode)
                # text = text.replace(' ','')
                texts.append(text)
                num_of_text = self.text_to_num(text)
                x.append(img_np)
                y.append(num_of_text)
            x = np.array(x)
            y = pad_sequences(y,maxlen=24,padding='post', truncating='post')
            yield x, y, texts
            
    def generator_of_ctc(self, 
                         batch_size=32,
                         input_shape=(32,480,1),
                         shuffle_text=True,
                         mode='test',
                         path='../corpus/address_mini.txt'):
        gen_xy = self.generator_of_xy(batch_size=batch_size,
                                      shuffle_text=shuffle_text,
                                      mode=mode,
                                      path=path)
        while True:
            x, y, texts = next(gen_xy)
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

    tti = TexttoImg(characters)
    print('The number of characters', tti.nclass)
    gen_train = tti.generator_of_ctc(batch_size=16,
                                     input_shape=(32,480,1),
                                     shuffle_text=True,
                                     mode='train',
                                     path='../corpus/address_mini.txt')
    for i in range(2):
        inputs, outputs = next(gen_train)
    
    for k,v in inputs.items():
        print(k, len(v))
    

