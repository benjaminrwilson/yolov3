# YOLOv3: An Incremental Improvement

### Paper

https://arxiv.org/pdf/1804.02767.pdf

### Installing + MS COCO Weights

```
python setup.py build develop
```

### Running on Images
In **detect.py** set **mode** to **images**. Set your images directory. Then run:

```
cd yolov3/demo/ && \
python detect.py
```

### Running on live video

```
cd yolov3/demo/ && \
python cam.py
```

### Evaluation

Evaluating the model on MS COCO is extremely simple. First, get all the necessary images and annotations with this script:

```
sh yolov3/tools/get_coco.sh
```

This will create a directory in your home directory called 'coco'. Next, create another directory called 'datasets' inside of the project, and softlink the 'coco' folder that we created.

```
mkdir datasets && \
ln -s ~/coco datasets/coco
```

To evaluate:

```
python yolov3/demos/eval.py
```

At 608 input resolution, this should achieve 56% @ AP50

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
