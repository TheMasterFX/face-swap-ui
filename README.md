# Face Swap UI
A browser interface based on Gradio for the Insightface face swap functionality

## Overview
The goal of this webui is to make it as easy as possible to create deepfakes using the Insightface framework. The UI provides a swap functionality for Images and Video.

It is still in very early development and it is the first time I'm working with the Gradio Framework, so things might not be implemented in a optimal way.

Feel free to add feature and pull requests.

This project was inspired by [roop-project](https://github.com/s0md3v/roop)

## Features
* Swap a single face in an image or a video
* Swap multiple faces in an image by just drag and drop the previous output to the input image
* Selecting the face to swap via a mouse click
* In video you can influence the face detection tolerance so only the selected face is swapped

## Installation
Clone this repo `git clone https://github.com/TheMasterFX/face-swap-ui.git` and install the required python packages `pip install -r requirement.txt`
It seems like Insightface has removed the *inswapper_128.onnx* so you have to download it manually and put it in face-swap-ui folder. 
A source might be [HERE](https://huggingface.co/deepinsight/inswapper/tree/main)