# -*- coding: utf-8 -*-
import flask
from flask import request, Flask, redirect, url_for, render_template

from models.AE_Model import AE_Model
from models.Combine_Model import InferenceModel
from options.AE_face import wholeOptions
from options.parts_combine import CombineOptions
import random
import numpy as np
import cv2
import jittor as jt
jt.flags.use_cuda = 1
import time

if __name__ == "__main__":
    app = Flask(__name__)
    mask = {}
    model = {}
    vector_part = {}
    width = 512
    refine = True
    part = {'eye1':(108,156,128),'eye2':(255,156,128),'nose':(182,232,160),'mouth':(169,301,192),'':(0,0,512)}
    opt = wholeOptions().parse(save=False)
    for key in  part.keys():
        opt.partial = key
        model[key] = AE_Model()
        model[key].initialize(opt)
        model[key].eval()
        mask[key] = cv2.cvtColor(cv2.imread('heat/' + key + '.jpg'), cv2.COLOR_RGB2GRAY).astype(np.float) / 255
        mask[key] = np.expand_dims(mask[key], axis=2)
        
    part_weight = {'eye1': 1,'eye2': 1,'nose': 1,'mouth': 1,'': 1}
    
    
    opt1 = CombineOptions().parse(save=False)
    opt1.nThreads = 1  
    opt1.batchSize = 1 
    #sex = 1
    sample_Num = 15
    shadow = {}
    combine_model = InferenceModel()
    combine_model.initialize(opt1)
    combine_model.eval()
    
    print('start sketch_img')  
    @app.route('/predict', methods=['GET', 'POST'])
    def predict():
        if request.method == 'POST':
            start=time.time()
            img = request.files['image'].read()
            print('read spend--',time.time()-start)
            #print(request.form.getlist('sex'))
            sex=int(request.form.getlist('sex')[0])
            random_ = random.randint(0, model[''].feature_list[sex].shape[0])
            img = np.fromstring(img, np.uint8)
            sketch = cv2.imdecode(img, flags=1)
            print('imdecode spend--',time.time()-start)
            for key in model.keys():
                loc = part[key]
                sketch_part = sketch[loc[1]:loc[1]+loc[2],loc[0]:loc[0]+loc[2],:]
                if key == '' and refine:
                    for key_p in model.keys():
                        if key_p!= '':
                           loc_p = part[key_p]
                           sketch_part[loc_p[1]:loc_p[1]+loc_p[2],loc_p[0]:loc_p[0]+loc_p[2],:] = 255
                if ((255-sketch_part).sum()==0):
                    shadow_, vector_part[key] = model[key].get_inter(sketch_part[:, :, 0],sample_Num,w_c = part_weight[key],random_=random_,sex=sex)
                else:
                    shadow_, vector_part[key] = model[key].get_inter(sketch_part[:, :, 0],sample_Num,w_c = part_weight[key],sex=sex)
    
                if key == '':
                    for key_p in model.keys():
                        if key_p!= '':
                            loc_p = part[key_p]
                            shadow_[loc_p[1]:loc_p[1]+loc_p[2],loc_p[0]:loc_p[0]+loc_p[2],:] = 255-(255-shadow_[loc_p[1]:loc_p[1]+loc_p[2],loc_p[0]:loc_p[0]+loc_p[2],:]) * (1-(1-mask[key_p])*0.2)
                shadow[key] = np.ones((width, width,1),dtype=np.uint8)*255
                shadow[key][loc[1]:loc[1]+loc[2],loc[0]:loc[0]+loc[2],:] = 255-(255-shadow_)* (1 - mask[key])
            print('get_inter before spend--',time.time()-start)
        
            shadow_, vector_part[key] = model[key].get_inter(sketch_part[:, :, 0],sample_Num,w_c = part_weight[key],sex=sex)
            print('get_inter model spend--',time.time()-start)
            generated=combine_model.inference(vector_part)
            ls=base64.b64encode(generated.tobytes())
            print('inference  spend--',time.time()-start)
            #print(generated)
        return ls
    app.run(threaded=True)
