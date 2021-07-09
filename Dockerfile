# build
# docker build -t paddleocr:gpu .

# run
# nvidia-docker run --name Ultra_light_OCR_No.4 -it paddleocr:gpu /bin/bash

# Version: 2.0.2
FROM registry.baidubce.com/paddlepaddle/paddle:2.0.2-gpu-cuda10.2-cudnn7

# PaddleOCR base on Python3.7
RUN pip3.7 install --upgrade pip -i https://mirror.baidu.com/pypi/simple

RUN git clone https://github.com/YuxinZou/Ultra_light_OCR_No.4.git /PaddleOCR

WORKDIR /PaddleOCR

RUN pip3.7 install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
