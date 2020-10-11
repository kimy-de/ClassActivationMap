# Class Activation Map(Pytorch)

Class Activation Map(CAM) was introduced in "[Learning Deep Features for Discriminative Localization" by B. Zhou et al., 2015](https://arxiv.org/abs/1512.04150). In this code, the CNN model is built based on a modified AlexNet with Global Average Pooling(GAP). 
# 1. Train + CAM image
```
python3 main.py --training y
```

# 2. Pretrained model + CAM image
```
python3 main.py --training n --model_path './cam_model.pth'
```

# 3. Custom datasets
First, please add your dataset in datasets.py.
```
python3 main.py --training True --data_path './your/path/'
```
<img width="500" alt="aa" src="https://user-images.githubusercontent.com/52735725/95685248-08f47c00-0bf7-11eb-9864-204d68703599.png">

