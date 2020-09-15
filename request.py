import argparse
import requests
import time
import cv2
import  numpy as np
import gevent
import base64
URL = 'http://127.0.0.1:5000/predict'

def predict_result(image_path,sex=2):
    start=time.time()
    f = open(image_path,'rb')
    ls_f=base64.b64encode(f.read())
    data = {'image':ls_f, 'sex':sex}
    print('spend--',time.time()-start)
    r = requests.post(URL,data=data)
    #print(r.text)
    print('spend--',time.time()-start)
    r =np.fromstring(base64.b64decode(r.text), np.uint8)
    r=r.reshape(512,512,3)
    print(r.shape)
    print('spend--',time.time()-start)
    print('spend--',time.time()-start)
    f.close()
    cv2.imwrite(str(int(time.time()))+'.jpg',cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
    
if __name__ =='__main__':
    #start=time.time()
    run_gevent_list = []
    for i in range(10):
        run_gevent_list.append(gevent.spawn(predict_result('hand-draw.jpg',sex=0)))
    gevent.joinall(run_gevent_list)
