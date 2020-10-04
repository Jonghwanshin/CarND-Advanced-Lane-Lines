# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Introduction
---

In this project, I implemented a software pipeline that detects lane lines in images using Python and OpenCV. This pipeline takes images from a video of driving scene on public road. Then, it finds each lane boundaries from each video frame. Finally, it returns a video file that visualizes the found lane and its curvature also the current estimated position from the center of the lane as shown below. Please refer `writeup.md` for detailed information.

![Lanes Image](./examples/example_output.jpg)
*Fig. 1: An Example Outfut of the project*

Installation
---

You need to install a python environment given in [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit).
In my case, creating conda environment given the above link doesn't work therefore I used a modified conda environment in [here](https://github.com/udacity/CarND-Term1-Starter-Kit/pull/119/commits).

To install conda environment, use following command to install and activate the conda environment.

for CPU environment:
```bash
conda env create -y environment.yml
conda activate carnd-term1
```

for GPU environment:
```bash
conda env create -y environment-gpu.yml
conda activate carnd-term1
```

Usage
---
Execute jupyter notebook with this command and open `advanced-lane-lines.ipynb` in the notebook and run cells.
```bash
jupyter notebook
```

License
---

This repository is under [MIT](https://choosealicense.com/licenses/mit/) license.