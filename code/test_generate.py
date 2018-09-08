# -*- coding: utf-8 -*-



from PIL import Image,ImageFont,ImageDraw
import numpy as np


class Text2Img():
    def __init__(self):
        
        self.bg = []
        img_bg = Image.open('../background/front.jpg')
        self.bg.append(img_bg)
        img_bg = Image.open('../background/back.jpg')
        self.bg.append(img_bg)
        

    def drawFont(self, txt):  
        font_size = 20
        font = ImageFont.truetype("../font/STFANGSO.TTF", font_size)  
    #    img = Image.new("RGB", (280, 32),(255,255,255))  
        img = Image.open('../background/front.jpg')
        img = img.crop((10,10,len(txt)*font_size + 10,42))
        print(img.size)
        draw = ImageDraw.Draw(img)  
        draw.text((0, 0), txt[:], font=font, fill=(0,0,0))  
        degree = np.random.randint(-4,4)
        img = img.rotate(degree, expand=False)
        img_np = np.array(img).astype(np.float32)
        noise = np.random.normal(0,0.5,img_np.shape)
        img_np = img_np + noise
        
        img = Image.fromarray(img_np.astype(np.uint8))
        img = img.resize((280,32))
        img = img.convert('L')
        img.save("../output/text.jpg")  
  
  
if __name__ == '__main__':  
    
    txt = '云南省文山壮族苗族自治州丘北县新店彝族乡蚌常村民委老龙树村小组'
    drawFont(txt[:14])  
