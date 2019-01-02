import subprocess

from setuptools import find_packages, setup

process = subprocess.Popen(["sh", "get_weights.sh"],
                           cwd="yolov3/tools/")
process.wait()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='yolov3',
      version='0.1',
      description='YOLOv3 implementation in PyTorch',
      url='https://github.com/benjaminrwilson/yolov3',
      author='Benjamin Wilson',
      license='MIT',
      install_requires=requirements,
      packages=find_packages(),
      zip_safe=False)
