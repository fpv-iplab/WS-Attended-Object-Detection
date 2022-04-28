# Weakly Supervised Attended Object Detection Using Gaze Data as Annotations
This is the official repository of the paper Weakly Supervised Attended Object Detection Using Gaze Data as Annotations

<b>Paper link: https://arxiv.org/abs/2204.07090</b> <br />
<b>Site link: https://iplab.dmi.unict.it/WS_OBJ_DET/</b> <br />


![alt text](./full_metod.gif)

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
