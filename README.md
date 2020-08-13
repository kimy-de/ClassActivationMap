# Class Activation Map

The CAM is built based on a modified AlexNet of pytorch.
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
