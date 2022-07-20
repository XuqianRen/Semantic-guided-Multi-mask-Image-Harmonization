from PIL import Image
import pilgram
import cv2 as cv
import numpy as np
import random
from scipy import signal
import os
import random
import numpy as np
import torch
import imgaug.augmenters as ia


def lab( comp):

    comp = np.array(comp)
    comp = cv.cvtColor(comp,cv.COLOR_RGB2LAB)
    (l, a, b) = cv.split(comp)
    
    lr = random.uniform(0.8,1.2)
    ar = random.uniform(0.8,1.2)
    br = random.uniform(0.8,1.2)
    l = l * lr
    a = a * ar
    b = b * br      

    l = np.clip(l, 0,255)
    a = np.clip(a, 0,255)
    b = np.clip(b, 0,255)
    comp = cv.merge([l, a, b])   
    comp = cv.cvtColor(comp.astype("uint8"),cv.COLOR_LAB2RGB) 
    comp = Image.fromarray(comp)
    return comp



class Perturb_Simulator:

    def __init__(self, gt_size ):
        self.gt_size = gt_size
        self.filters =[  pilgram._1977,
           pilgram.brannan,
           pilgram.brooklyn,
           pilgram.clarendon,
           pilgram.earlybird,
           pilgram.gingham,
           pilgram.hudson,
           pilgram.inkwell,
           pilgram.kelvin,
           pilgram.lark,
           pilgram.lofi,
           pilgram.maven,
           pilgram.mayfair,
           pilgram.moon,
           pilgram.nashville,
           pilgram.reyes,
           pilgram.rise,
           pilgram.slumber,
           pilgram.stinson,
           pilgram.valencia,
           pilgram.walden,
           pilgram.willow,
           pilgram.xpro2         
             ]


        self.other_filters = [
                pilgram.css.contrast,
                pilgram.css.grayscale,#
                pilgram.css.hue_rotate,
                pilgram.css.saturate,
                pilgram.css.sepia, #
                lab
              ]
        self.BLUR_TEMPLATES = {
            'denoise':
            ia.OneOf([
                ia.AdditiveGaussianNoise(scale=(8, 10), per_channel=True),
                ia.AdditiveLaplaceNoise(scale=(8, 10), per_channel=True),
                ia.AdditivePoissonNoise(lam=(8, 10), per_channel=True),
            ]),
            'deblur':
            ia.OneOf([
                ia.MotionBlur(k=(3, 3)),
                ia.GaussianBlur((1.0, 1.0)),
            ]),
            'jpeg':
            ia.JpegCompression(compression=(5, 5)),
        }
        self.rand_deg_list = ['denoise','deblur','jpeg']

        


    def perturb(self, real, mask):
        # 选择multi mask区域
        idx1 = np.unique(mask)

        classes = []
        for id in idx1:
            if id == 0:
                continue
            classes.append(id)  

        choosen_class = np.random.choice(classes, int(len(classes) * 0.2) + 1, replace=True) # int(len(classes) * 0.2) + 
        choosen_class = np.unique(choosen_class)

        #对每个mask分别施加一种perturb方法
        final_mask = np.zeros(mask.shape)
        real = cv.cvtColor(real, cv.COLOR_BGR2RGB)
        real = Image.fromarray(real)
        fake_list = list()
        mask_list = list()

        for changed_class in choosen_class:   
            
            temp_mask = np.zeros(mask.shape)
            temp_mask[mask == changed_class] = 1            
            final_mask[temp_mask == 1] = 1
            temp_mask = np.expand_dims(temp_mask, axis=2)
        
            # 滤镜
            filter1 = np.random.choice(self.filters)
            fake = filter1(real)
            # fake = real
            
            q = np.random.random(1)
            if q>0.5:
                filter2 = np.random.choice(self.other_filters)
                fake = filter2(fake) 
            fake = np.array(fake)
            fake = cv.cvtColor(fake, cv.COLOR_RGB2LAB)


            fake_list.append(fake)
            mask_list.append(temp_mask)

        real = np.array(real)
        real = cv.cvtColor(real, cv.COLOR_RGB2LAB)
        comp = real
        for i in range(len(fake_list)):
            comp = fake_list[i]*mask_list[i] + comp*(1-mask_list[i])

    

        mask = np.expand_dims(final_mask,axis=2)
        m = np.random.random(1)
        if m>0.95:
        # if m:
            deg =  np.random.choice(self.rand_deg_list)
            real = self.BLUR_TEMPLATES[deg].augment_image(real)
            comp = comp*mask+real*(1-mask)

        real = cv.resize(real,(self.gt_size,self.gt_size))
        comp = cv.resize(comp,(self.gt_size,self.gt_size))
        mask = cv.resize(mask,(self.gt_size,self.gt_size))
              
        real = real.astype(np.float32)
        comp = comp.astype(np.float32)
        mask = mask.astype(np.float32)
        
        
        
        return real, comp, mask

if __name__ == '__main__':
    simuator = Perturb_Simulator(256)
    gt_img = cv.imread('/home/dzh/data2/data2/BasicSR/datasets/HLIP/test/images/36_453991.jpg')
    label_img = cv.imread('/home/dzh/data2/data2/BasicSR/datasets/HLIP/test/labels/36_453991.png',0)

    real, comp, mask = simuator.perturb(gt_img, label_img)
    comp = cv.cvtColor(comp.astype("uint8"), cv.COLOR_LAB2BGR)
    real = cv.cvtColor(real.astype("uint8"), cv.COLOR_LAB2BGR)
    mask = mask[:,:,None].repeat(3,axis=2)*255
    gt_img = cv.resize(gt_img,(256,256))
    new = np.concatenate([gt_img, comp,mask],1)
    cv.imwrite('./instragram.png',new)