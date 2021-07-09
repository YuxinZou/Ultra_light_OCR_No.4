## Requirements
- Linux
- Python 3.7
- paddlepaddle-gpu 2.0.2+

## 我们在以下环境中进行过测试:
- OS: Ubuntu 16.04.6 LTS
- CUDA: 10.2
- CUDNN: 7.6.5

## 1. docker部署
```shell script
#切换至Dockerfile目录
cd Ultra_light_OCR_No.4
```

```shell script
#生成镜像
docker build -t paddleocr:gpu .
```

```shell script
#运行镜像
nvidia-docker run --name Ultra_light_OCR_No.4 -it paddleocr:gpu /bin/bash
```

## 2. 数据集放置结构
```
data
├── trainval
│   ├── TrainImages
│   │   ├── Train_000000.jpg
│   │   ├── Train_000001.jpg
|   |   ├── ......
│   ├── LabelTrain.txt
│── testA
│   │   ├── TestAImages
│── testB
│   │   ├── TestBImages
```

## 3. 训练指令
- step1
```shell script
python3.7 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/rec/new_baseline/step1.yml
```

- step2:
```shell script
python3.7 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/rec/new_baseline/step2.yml -o Global.checkpoints=checkpoint/step1/iter_epoch_1700
```

## 4. 预测testB数据集
```shell script
# 静态图模型存储位置"./checkpoint/upload_model/infer"
python3.7 tools/infer/predict_rec_media_smart.py \
			--rec_config=./configs/rec/new_baseline/step2.yml \
			--rec_model_dir=./checkpoint/upload_model/infer \
			--image_dir=./data/testB/TestBImages \
			--save_res_path=./output/rec/testB.txt  
```
