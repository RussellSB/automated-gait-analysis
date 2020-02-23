# Automated-Gait-Analysis
A human pose estimation system for videos, that aims to extract features describing a gait (walk), with respect to kinematics.
This is my final year project for my BSc. Honours in Artificial Intelligence. A work in progress.

![alt-text](https://github.com/RussellSB/Automated-Gait-Analysis/blob/master/demo/example.gif)
First step: Extract keypoints from synchronized video sequences using Pre-trained AI models: Object detector YOLOv3, and Pose estimator AlphaPose. Keypoints, represent the point of a joint [x,y]. Pose, is a list of 17 keypoints. Data, is a list of n poses, where n is the number of frames in the video. The concepts are stored in a json.

<p align="center">
  <img src="https://github.com/RussellSB/Automated-Gait-Analysis/blob/master/demo/example2.gif">
</p>
Second step: Extract raw angle kinematics

<p align="center">
  <img src="https://github.com/RussellSB/automated-gait-analysis/blob/master/demo/example3_1.png">
</p>
<p align="center">
  <img src="https://github.com/RussellSB/automated-gait-analysis/blob/master/demo/example3_2.png">
</p>
Third step: Process kinematics. The pipeline for following follows: gap filling, smoothing, gait cycle slicing, resampling and finally averaging. In the first picture above we see an example of smoothing whereas in the second we see the average gait cycle of knee flexion taken from a single capture.
