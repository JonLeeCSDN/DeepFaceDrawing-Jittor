# DeepFaceDrawing: Deep Generation of Face Images from Sketches(Flask version)
## Prerequisites

1. System
  ```
　- Ubuntu 16.04 or later
　- NVIDIA GPU + CUDA 9.2(later  version also can  be sucess)
    - python 3.7
  ```
2. install require Packages
  ```
  sh install.sh
  ```
  3. rename a figure:
```
mv heat/bg.jpg heat/.jpg
```

  4. prepare model file

Please download the pre-trained model<a href="https://pan.baidu.com/s/1f1S9t4T5X5J0CDZ7AqTfMg 
" target="_blank">[Baidu(Password:wiu9)]</a> and put those under 'Param'.

## How to use
  ```
  server:
  python3.7 inference.py
  client:
   python3.7 request.py
  ```

