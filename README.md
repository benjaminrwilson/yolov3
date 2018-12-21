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

### Get the MS COCO Weights

```
cd yolov3/tools/ && \
sh get_models.sh
```

### Running on Images
Boston_Terrier_male.jpg
In **detect.py** set **mode** to **images**. Add your images to the **images** directory. Then run:

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
