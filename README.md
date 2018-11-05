# YOLOv3

https://arxiv.org/pdf/1804.02767.pdf

<img src="https://github.com/benjaminrwilson/yolov3/blob/master/results/Boston_Terrier_male.jpg" width="300">  

### Prerequisites

```
opencv_python 3.4.2.17
numpy 1.15.1
torch 0.4.1
torchvision 0.2.1
```

### Installing

```
pip install -r requirements.txt
```

### Get the COCO Weights

```
sh get_yolov3_weights.sh
```

in the weights directory

### Running on Images

Add images to the **images** directory, set **mode** to **images** then run:

```
python detect.py
```

### Running on live video

In **detect.py** set **mode** to **cam**. Then run:

```
python detect.py
```

## Future Improvements

- [ ] Cuda Support
- [ ] Training Support
