# HFA-PANet-for-MCD
The PR2025 paper, "Hierarchical Feature Alignment-based Progressive Addition Network for Multimodal Change Detection," has been published in ***Pattern Recognition***.
This repository contains a PyTorch implementation of HFA-PANet.

## Outline

<ul>

 <li>Introduction</li>

  <li>Requirements</li>


  <li>Installation</li>

  <li>Usage</li>

  <li>Parameters</li>

  <li>Results</li>

  <li>Example</li>

  <li>References</li>

</ul>

## Introduction

Multimodal change detection involves identifying changes between images captured at different times and using different sensors (e.g., optical and SAR). Although some patch-level MCD methods have been reported, there are still few studies on large-scale image-level MCD methods. The main challenge of MCD is that it is more difficult to capture the differences between different modality images than heterogeneous BTIs. To address the challenge, this paper proposes a novel non-Siamese Hierarchical Feature Alignment-based Progressive Addition Network (HFA-PANet) for MCD. In the proposed HFA-PANet, two novel modules are devised to elevate the difference features of multimodal BTIs, thereby improving its change extraction capability. The framework of the proposed HFA-PANet is presented as follows:
![Framework of our proposed HFA-PANet)](https://github.com/TongfeiLiu/HFA-PANet-for-MCD/blob/main/Fig/Fig2-Framework.png)

## Requirements
<ul>

  <li>Python=3.8</li>

  <li>PyTorch=2.4.0 </li>

  <li>tqdm=4.67</li>

  <li>NumPy=1.24.0</li>

  <li>torchvision=0.19.1
</li>

</ul>

## Installation

### 1. Clone the repository:

git clone https://github.com/TongfeiLiu/HFA-PANet-for-MCD.

cd HFA-PANet

### 2. Set up a virtual environment(Optional, but not recommended)

python -m venv venv

source venv/bin/activate  

### 3. Install the required packages:

pip install -r requirements.txt

## Usage

### 1. Prepare your data:

* First-time image: e.g., SAR image.
  
* Second-time image: e.g., optical image.
  
* Reference ground truth: Ground truth change map for evaluation
  
### 2. Modify the script or provide command-line arguments:

* Update the script with the path to the image and other parameters
  
### 3. Run the script:

* python Edition.py (This will give the results of training and testing)

### 4. Please set up your folder like this:

rootdir\

data\dataset\

MTWHU\

sar\

|---- image1.png

|---- image2.png

|---- image3.png

|---- .... .png

opt\

|---- image1.png

|---- image2.png

|---- image3.png

|---- .... .png

lable\

|---- label1.png

|---- label2.png

|---- label3.png

|---- label.....png

## Result

After running the script, you will obtain:

Binary Change Maps: Thresholded maps showing detected changes.

Performance Metrics: A txt file containing Overall Accuracy, recall, CIOU, and F1 score, etc.

## Citation
If you find our work useful for your research, please consider citing our paper:
```
@article{PR2025HFA-PANet,
title = {Hierarchical Feature Alignment-based Progressive Addition Network for Multimodal Change Detection},
journal = {Pattern Recognition},
volume = {162},
pages = {111355},
year = {2025},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.111355},
url = {https://www.sciencedirect.com/science/article/pii/S0031320325000159},
author = {Tongfei Liu and Yan Pu and Tao Lei and Jianjian Xu and Maoguo Gong and Lifeng He and Asoke K. Nandi},
keywords = {Multimodal change detection, Heterogeneous change detection, Heterogeneous images, Feature alignment, Multiple kernel maximum mean discrepancy}
}
```

## Contact us 

If you have any problems when running the code, please do not hesitate to contact us. Thanks.  
E-mail: liutongfei_home@hotmail.com

Date: June 4, 2025  
