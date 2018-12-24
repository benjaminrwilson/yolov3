# YOLOv3: An Incremental Improvement

<p align="center">
  <img src="https://github.com/benjaminrwilson/yolov3/blob/master/yolov3/results/dogs.jpg" width="800">  
</p>

### Paper

https://arxiv.org/pdf/1804.02767.pdf

### Installing

```
python setup.py build develop
```

### Download the MS COCO Weights

```
cd yolov3/tools/ && \
sh get_models.sh
```

### Running on Images
In **detect.py** set **mode** to **images**. Set your images directory. Then run:

```
cd yolov3/demo/ && \
python detect.py
```

### Running on live video

In **detect.py** set **mode** to **cam**. Then run:

```
cd yolov3/demo/ && \
python detect.py
```

### Simple API

```
import os
from yolov3.demo.detect import Config, get_model

home = os.path.expanduser("~")
config = os.path.join(home, "code/yolov3/yolov3/config/yolov3.cfg")
weights = os.path.join(home, ".torch/yolov3/yolov3.weights")
nms = .45
obj = .5
size = 416

config = Config(config, weights, nms, obj, size)
model = get_model(config)

image_path = "path_to_your_image.jpg"
detections = model.detect(image_path)
```

## Future Improvements

- [x] Cuda Support
- [ ] Training Support

## References

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
