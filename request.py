import argparse
import requests
import time
import cv2
import  numpy as np
URL = 'http://127.0.0.1:5000/predict'

def predict_result(image_path,sex=1):
    start=time.time()
    img = open(image_path,'rb').read()
    print('spend--',time.time()-start)
    msg = {'image':img}
    data={'sex':sex}
    r = requests.post(URL,files=msg,data=data)
    print('post spend--',time.time()-start)
    #print(r.url)
    r = r.json()
    r=np.array(r).astype(np.uint8)
    #print('sucess',r)
    print('spend--',time.time()-start)
    cv2.imwrite('result/'+str(int(time.time()))+'.jpg',cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
    #print('save  sucess')
    print('spend--',time.time()-start)
if __name__ =='__main__':
    predict_result('XXXX.jpg',sex=1)
