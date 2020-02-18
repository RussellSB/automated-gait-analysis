#==================================================================================
#                               VISUALIZER
#----------------------------------------------------------------------------------
#                           Input: JSON, Output: Plots
#               Visualizes saved graph structure of poses, as well as
#               saved raw kinematics, and processed kinematics
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

#==================================================================================
#                                   Methods
#==================================================================================
# PLots and saves every pose frame of the video
def plot_poses(data, dim, path, limit):
    i = 1
    for pose in data:
        fig, ax = plt.subplots()
        ax.set(xlim=(0, dim[0]), ylim=(0, dim[1]))  # setting width and height of plot

        for cm_ind, jp in zip(colormap_index, joint_pairs):
            joint1 = pose[jp[0]]
            joint2 = pose[jp[1]]
            if (joint1 > [-1, -1] and joint2 > [-1, -1]):
                x = [joint1[0], joint2[0]]
                y = [joint1[1], joint2[1]]
                ax.plot(x, y, linewidth=3.0, alpha=0.7, color=plt.cm.cool(cm_ind))
                ax.scatter(x, y, s=20)
        filename = path + str(i) + '.png'
        plt.savefig(filename)
        if(i == limit): break
        i += 1

# Makes a collection of figures out of what is described in the jsonFile
def jsonPose_to_plots(jsonFile, path):
    with open(jsonFile, 'r') as f:
        jsonPose = json.load(f)

    lenF = jsonPose[0]['lenF']
    lenS = jsonPose[0]['lenS']
    limit = min(lenF, lenS)

    dataS = jsonPose[0]['dataS']
    dimS = jsonPose[0]['dimS']
    path1 = path + jsonPose[0]['id'] + '-S/'
    plot_poses(dataS, dimS, path1, limit)

    dataF = jsonPose[0]['dataF']
    dimF = jsonPose[0]['dimF']
    path2 = path + jsonPose[0]['id'] + '-F/'
    plot_poses(dataF, dimF, path2, limit)

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

# TODO: Method for json Angles, calls plotting functions
# def jsonAngles_to_pics(jsonFile):

# path = '../Test/GIF/'
# jsonPose_to_plots('test.json', path)

with open('test_anglesFix.json', 'r') as f:
    jsonAngles = json.load(f)

raw_angles = jsonAngles[0]
knee_FlexExt = raw_angles['knee_FlexExt']
hip_FlextExt = raw_angles['hip_FlexExt']

plot_angles(knee_FlexExt, 'Knee Flexion/Extension', (-20, 80))
plot_angles(hip_FlextExt, 'Hip Flexion/Extension', (-20, 60))

