# Automated-Gait-Analysis
A human pose estimation system for videos, that aims to extract features describing a gait (walk), with respect to kinematics.
This is my final year project for my BSc. Honours in Artificial Intelligence. A work in progress.

![alt-text](https://github.com/RussellSB/Automated-Gait-Analysis/blob/master/demo/example.gif)
First step: Extract keypoints from synchronized video sequences using Pre-trained AI models: Object detector YOLOv3, and Pose estimator AlphaPose. Keypoints, represent the point of a joint [x,y]. Pose, is a list of 17 keypoints. Data, is a list of n poses, where n is the number of frames in the video. The concepts are stored in a json.
