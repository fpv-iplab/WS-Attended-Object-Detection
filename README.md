# Weakly Supervised Attended Object Detection Using Gaze Data as Annotations
This is the official repository of the paper Weakly Supervised Attended Object Detection Using Gaze Data as Annotations

<b>Project Web Page: https://iplab.dmi.unict.it/WS_OBJ_DET/</b> <br />


![alt text](./full_metod.gif)

## Requirements
<ul>
<li>Linux or macOS with Python ≥ 3.6</li>
<li>PyTorch ≥ 1.8 and torchvision that matches the PyTorch installation. Install them together at pytorch.org to make sure of this</li>
<li>OpenCV is optional but needed by demo and visualization</li>
</ul>
This tool use Detectron2. Please see <a href="https://github.com/facebookresearch/detectron2">installation instructions </a>.


## (DIR) Detection(Fully-Supervised)
Contains:
The scripts used for training the Detectron 2-based network.
The inference script useful to obtain the prediction s of the boxes of all the objects present in the frame.
The script used to calculate the mAPs by filtering the box detections using the 2D gaze.

## (DIR) WS sliding window
Contains:  
The script used to train the ResNet-18 based network with 300x300 pixel patches clipped around the gaze points.
The script that through the sliding window approach obtains segmentation masks where each color represents the color of the patch. 

## (DIR) WS Fully connected
Contains:  
The script used to make the changes and finetuning to the previously trained ResNet-18 network. The loss function present is the kullback-leibler divergence loss.
The script that allows you to transform probability distributions into segmentation masks and calculate mAPs



## Cite Weakly Supervised Attended Object Detection Using Gaze

```
@inproceedings{mazzamuto21wlsgaze,
  title = {Weakly Supervised Attended Object Detection Using Gaze Data as Annotations},
  author = {Michele Mazzamuto and Francesco Ragusa and Antonino Furnari and Giovanni Signorello and Giovanni Maria Farinella},
  year = {2021}
  booktitle = {International Conference on Image Analysis and Processing (ICIAP), May 23-27 2022},
}

```
