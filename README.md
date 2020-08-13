# Class Activation Map(Pytorch)

Class Activation Map(CAM) was introduced in "[Learning Deep Features for Discriminative Localization" by B. Zhou et al., 2015](https://arxiv.org/abs/1512.04150). In this code, our model is built based on a modified AlexNet with a Global Average Pooling(GAP). 
# 1. Train + CAM image
```
python3 main.py --training True
```

# 2. Pretrained model + CAM image
```
python3 main.py --training False --model_path './cam_model.pth'
```

# 3. Custom datasets
```
python3 main.py --training True --data_path './your/path/'
```
![Unknown-5](https://user-images.githubusercontent.com/52735725/90183047-29e55000-ddb3-11ea-91d6-7789afccbcd3.png)
