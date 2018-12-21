# YOLOv3

https://arxiv.org/pdf/1804.02767.pdf

<img src="https://github.com/benjaminrwilson/yolov3/blob/master/results/Boston_Terrier_male.jpg" width="300">  

### Installing

```
python setup.py build develop
```

### Get the MS COCO Weights

```
cd tools/ && \
sh get_models.sh
```

### Running on Images

In **detect.py** set **mode** to **images**. Add your images to the **images** directory. Then run:

```
cd demo/ && \
python detect.py
```

### Running on live video

In **detect.py** set **mode** to **cam**. Then run:

```
cd demo/ && \
python detect.py
```

## Future Improvements

- [x] Cuda Support
- [ ] Training Support
