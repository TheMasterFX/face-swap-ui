# Face Swap UI
A browser interface based on Gradio for the Insightface face swap functionality

https://github.com/TheMasterFX/face-swap-ui/assets/12451336/7adde181-ac2d-40bc-93fe-15000b9706c2

## Overview
The goal of this webui is to make it as easy as possible to create deepfakes using the Insightface framework. The UI provides a swap functionality for Images and Video.

It is still in very early development and it is the first time I'm working with the Gradio Framework, so things might not be implemented in a optimal way. 

The UI is not perfect at this point, but it works at least.

Feel free to add feature and pull requests.

This project was inspired by [roop-project](https://github.com/s0md3v/roop)

## Features
* Swap a single face in an image or a video
* Swap multiple faces in an image by just drag and drop the previous output to the input image
* Selecting the face to swap via a mouse click
* In video you can influence the face detection tolerance so only the selected face is swapped

## Missing Features
- [ ] GPU Support
- [ ] Keep audio from input video
- [ ] Swap multiple faces in one go

## Installation and Running
Clone this repo `git clone https://github.com/TheMasterFX/face-swap-ui.git` and install the required python packages `pip install -r requirement.txt`
It seems like Insightface has removed the *inswapper_128.onnx* so you have to download it manually and put it in face-swap-ui folder. 
A source might be [HERE](https://huggingface.co/deepinsight/inswapper/tree/main).
Ensure you have installed [ffmpeg](https://ffmpeg.org/) and it is in your PATH envronment variable.

I recommend creating a venv e.g.
```
$ cd face-swap-ui
$ python -m venv .\venv
$ .\venv\Scripts\Activate.ps1
$ pip install -r requirement.txt
$ python .\face_swap_ui.py
```
After the starting face-swap-ui use your browser and navigate to http://127.0.0.1:7860

## Hints
* The more faces in the image the longer it takes to process
* The smaller your input video resolution is the faster it processes


