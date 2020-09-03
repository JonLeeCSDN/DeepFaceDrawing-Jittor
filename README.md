# DeepFaceDrawing: Deep Generation of Face Images from Sketches
## Prerequisites

1. System

　- Ubuntu 16.04 or later

　- NVIDIA GPU + CUDA 9.2(later also can  be sucess) 

2. Software

　- Python 3.7

　- Jittor. More details in <a href="https://github.com/Jittor/Jittor" target="_blank">Jittor</a>

  ```
  sudo apt install python3.7-dev libomp-dev

  sudo python3.7 -m pip install git+https://github.com/Jittor/jittor.git

  python3.7 -m jittor.test.test_example
  ```

　- Packages

  ```
  sh install.sh
  ```

## How to use

Drawing sketch using DeepFaceDrawing GUI. Please download the pre-trained model<a href="https://pan.baidu.com/s/1f1S9t4T5X5J0CDZ7AqTfMg 
" target="_blank">[Baidu(Password:wiu9)]</a> and put those under 'Param'.

  ```
  server:
  python3.7 inference.py
  agent:
   python3.7 request.py
  ```

