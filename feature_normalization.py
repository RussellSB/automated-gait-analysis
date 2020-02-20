#==================================================================================
#                               FEATURE_NORMALIZATION
#----------------------------------------------------------------------------------
#                    Input: Pose Json and Raw angles, Output: Gait Cycle graphs
#               Given a JSON describing angles of joints throughout a walk,
#               Smoothens kinematics and averages to one standard gait cycle.
#----------------------------------------------------------------------------------
#==================================================================================
#==================================================================================
#                                   Imports
#==================================================================================
import numpy as np
import matplotlib.pyplot as plt
import json

#==================================================================================
#                                   Constants
#==================================================================================
joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                   [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]]
colormap_index = np.linspace(0, 1, len(joint_pairs))

ptID = {
    'nose': 0,
    'eye_L': 1,'eye_R': 2,
    'ear_L': 3,'ear_R': 4,
    'shoulder_L': 5, 'shoulder_R': 6,
    'elbow_L': 7, 'elbow_R': 8,
    'wrist_L': 9, 'wrist_R': 10,
    'hip_L': 11, 'hip_R': 12,
    'knee_L': 13, 'knee_R': 14,
    'ankle_L': 15, 'ankle_R': 16
}

#==================================================================================
#                                   Methods
#==================================================================================
# Plots left and right kinematics
def plot_angles(angleList, title, yrange):
    red = "#FF4A7E"
    blue = "#72B6E9"

    leftMax = len(angleList[0])
    rightMax = len(angleList[1])
    xmax = max(leftMax, rightMax)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Frame (count)')
    ax.set_ylabel(r"${\Theta}$ (degrees)")
    ax.set(xlim=(0, xmax), ylim=(yrange[0], yrange[1]))

    leftAngles = angleList[0]
    rightAngles = angleList[1]

    ax.plot(leftAngles, color=red)
    ax.plot(rightAngles, color=blue)

    plt.show()

# exponential moving average
def smooth(angle_list, weight):  # Weight between 0 and 1
    last = angle_list[0]  # First value in the plot (first timestep)
    smoothed = []
    for angle in angle_list:
        smoothed_val = last * weight + (1 - weight) * angle  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

#TODO: Try out other smoothing methods, focus on gait cycle extraction
def smoothLR(angles_list, weight):
    angles_L = angles_list[0]
    angles_R = angles_list[1]
    smooth_L = smooth(angles_L, weight)
    smooth_R = smooth(angles_R, weight)
    smoothed_LR = [smooth_L, smooth_R]

    return smoothed_LR

with open('test_angles3.json', 'r') as f:
    jsonAngles = json.load(f)

raw_angles = jsonAngles[0]

knee_FlexExt = raw_angles['knee_FlexExt']
hip_FlexExt = raw_angles['hip_FlexExt']
knee_AbdAdd = raw_angles['knee_AbdAdd']
hip_AbdAdd = raw_angles['hip_AbdAdd']

weight = 0.8
knee_FlexExt1 = smoothLR(knee_FlexExt, weight)
hip_FlexExt1 = smoothLR(hip_FlexExt, weight)
knee_AbdAdd1 = smoothLR(knee_AbdAdd, weight)
hip_AbdAdd1 = smoothLR(hip_AbdAdd, weight)

plot_angles(knee_FlexExt, 'Knee Flexion/Extension', (-20, 80))
plot_angles(knee_FlexExt1, 'Knee Flexion/Extension', (-20, 80))
plot_angles(hip_FlexExt, 'Hip Flexion/Extension', (-20, 60))
plot_angles(hip_FlexExt1, 'Hip Flexion/Extension', (-20, 60))
plot_angles(knee_AbdAdd, 'Knee Abduction/Adduction', (-20, 20))
plot_angles(knee_AbdAdd1, 'Knee Abduction/Adduction', (-20, 20)) #TODO: Cater for gaps
plot_angles(hip_AbdAdd, 'Hip Abduction/Adduction', (-30, 30))
plot_angles(hip_AbdAdd1, 'Hip Abduction/Adduction', (-30, 30))