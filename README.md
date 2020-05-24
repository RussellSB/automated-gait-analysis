# Automated-Gait-Analysis
A human pose estimation system for videos, that aims to extract features describing a gait (walk) and deploy classifiers to detect abnormalities and more, with respect to kinematics.
This is my final year project for my BSc. Honours in Artificial Intelligence.

![alt-text](https://github.com/RussellSB/Automated-Gait-Analysis/blob/master/demo/example.gif)
First step: Extract keypoints from synchronized video sequences using Pre-trained AI models: Object detector YOLOv3, and Pose estimator AlphaPose. Keypoints, represent the point of a joint [x,y]. Pose, is a list of 17 keypoints. Data, is a list of n poses, where n is the number of frames in the video. The concepts are stored in a json.

<p align="center">
  <img src="https://github.com/RussellSB/Automated-Gait-Analysis/blob/master/demo/example2.gif">
</p>
Second step: Extract raw angle kinematics. As demonstrated above knee flexion and extension is computed using the side view. In general hip flexion and extension is computed using the side view also, while hip and knee abduction/adduction are computed using the front view.

<p align="center">
  <img src="https://github.com/RussellSB/automated-gait-analysis/blob/master/demo/example3_1.png">
</p>
<p align="center">
  <img src="https://github.com/RussellSB/automated-gait-analysis/blob/master/demo/example3_2.png">
</p>
<p align="center">
  <img src="https://github.com/RussellSB/automated-gait-analysis/blob/master/demo/example3_3.png">
</p>
Third step: Process kinematics. The processing pipeline follows: gap filling, smoothing, gait cycle slicing, resampling and finally averaging. Demonstrating above, we see a smoothed gait cycle, an average of all gait cycles in one capture, and an average of all gait cycles in six captures (three walking left ot right and another three walking right to left).

