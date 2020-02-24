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

red = "#FF4A7E"
blue = "#72B6E9"
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

# PLots and saves every leg pose frame of the video
def plot_legs(data, dim, path):
    i = 1
    for pose in data:
        fig, ax = plt.subplots()
        ax.set(xlim=(0, dim[0]), ylim=(0, dim[1]))  # setting width and height of plot

        #Left
        x, y =[], []
        hip_L = pose[ptID['hip_L']]
        if(hip_L != [-1,-1]):
            x.append(hip_L[0])
            y.append(hip_L[1])
        knee_L = pose[ptID['knee_L']]
        if(knee_L != [-1,-1]):
            x.append(knee_L[0])
            y.append(knee_L[1])
        ankle_L = pose[ptID['ankle_L']]
        if(ankle_L != [-1,-1]):
            x.append(ankle_L[0])
            y.append(ankle_L[1])
        ax.scatter(x, y, s=20, color=red)
        ax.plot(x, y, color=red)

        #Right
        x, y = [], []
        hip_R = pose[ptID['hip_R']]
        if(hip_R != [-1,-1]):
            x.append(hip_R[0])
            y.append(hip_R[1])
        knee_R = pose[ptID['knee_R']]
        if(knee_R != [-1,-1]):
            x.append(knee_R[0])
            y.append(knee_R[1])
        ankle_R = pose[ptID['ankle_R']]
        if(ankle_R != [-1,-1]):
            x.append(ankle_R[0])
            y.append(ankle_R[1])
        ax.scatter(x, y, s=20, color=blue)
        ax.plot(x, y, color=blue)

        filename = path + str(i) + '.svg'
        plt.savefig(filename)
        i += 1

# Makes a collection of figures out of what is described in the jsonFile
def jsonLegs_to_plots(jsonFile, path):
    with open(jsonFile, 'r') as f:
        jsonPose = json.load(f)

    dataS = jsonPose[0]['dataS']
    dimS = jsonPose[0]['dimS']
    path1 = path + 'legs' + '-S/'
    plot_legs(dataS, dimS, path1)

    dataF = jsonPose[0]['dataF']
    dimF = jsonPose[0]['dimF']
    path2 = path + 'legs' + '-F/'
    plot_legs(dataF, dimF, path2)

# Plots left and right kinematics
def plot_angles(angleList, title, yrange):
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

# Plots left and right kinematics frame by frame and saves
def plot_angles2(angleList, title, yrange, path):
    leftMax = len(angleList[0])
    rightMax = len(angleList[1])
    xmax = max(leftMax, rightMax)

    leftTemp = []
    rightTemp = []

    for i in range(0, xmax):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel('Frame (count)')
        ax.set_ylabel(r"${\Theta}$ (degrees)")
        ax.set(xlim=(0, xmax), ylim=(yrange[0], yrange[1]))

        leftTemp.append(angleList[0][i])
        rightTemp.append(angleList[1][i])

        ax.plot(leftTemp, color=red)
        ax.plot(rightTemp, color=blue)
        filename = path + str(i) + '.svg'
        plt.savefig(filename)

path = '../Test2/GIF/'
jsonLegs_to_plots('../Test2/test.json', path)

#with open('../Test2/test_angles.json', 'r') as f:
#    jsonAngles = json.load(f)

#raw_angles = jsonAngles[0]
#knee_FlexExt = raw_angles['knee_FlexExt']
#hip_FlextExt = raw_angles['hip_FlexExt']
#knee_AbdAdd = raw_angles['knee_AbdAdd']
#hip_AbdAdd = raw_angles['hip_AbdAdd']

#path = '../Test2/GIF/'
#plot_angles2(knee_FlexExt, 'Knee Flexion/Extension', (-20, 80), path + 'knee_FlexExt/')
#plot_angles2(hip_FlextExt, 'Hip Flexion/Extension', (-20, 60), path + 'hip_FlexExt/')
#plot_angles2(knee_AbdAdd, 'Knee Abduction/Adduction', (-20, 20), path + 'knee_AbdAdd/')
#plot_angles2(hip_AbdAdd, 'Hip Abduction/Adduction', (-30, 30), path + 'hip_AbdAdd/')

