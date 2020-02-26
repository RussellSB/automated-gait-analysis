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
import io
from PIL import Image

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
def plot_poses(data, dim, filename):
    i = 1
    ims = [] # List of images for gif
    buf = io.BytesIO()

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

        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        ims.append(im)
        #im.show()

        if(i == 5): break
        i += 1
    im.save(filename, save_all=True, append_images=ims, duration=40, loop=0)
    buf.close()

# Makes a collection of figures out of what is described in the jsonFile
def jsonPose_to_plots(poseFile, outpath):
    with open(poseFile, 'r') as f:
        jsonPose = json.load(f)

    dataS = jsonPose[0]['dataS']
    dimS = jsonPose[0]['dimS']
    #path1 = outpath + jsonPose[0]['capId'] + '-S.gif'
    plot_poses(dataS, dimS, 'test-S.gif')

    dataF = jsonPose[0]['dataF']
    dimF = jsonPose[0]['dimF']
    #path2 = outpath + jsonPose[0]['capId'] + '-F.gif'
    plot_poses(dataF, dimF, 'test-F.gif')

# PLots and saves every leg pose frame of the video
def plot_legs(data, dim, outpath):
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

        filename = outpath + str(i) + '.svg'
        plt.savefig(filename)
        i += 1

# Saves leg plots of front and forward videos
def jsonLegs_to_plots(poseFile, outpath):
    with open(poseFile, 'r') as f:
        jsonPose = json.load(f)

    dataS = jsonPose[0]['dataS']
    dimS = jsonPose[0]['dimS']
    path1 = outpath + 'legs' + '-S/'
    plot_legs(dataS, dimS, path1)

    dataF = jsonPose[0]['dataF']
    dimF = jsonPose[0]['dimF']
    path2 = outpath + 'legs' + '-F/'
    plot_legs(dataF, dimF, path2)

# Saves left and right kinematics frame by frame
def save_angles(angleList, title, yrange, outpath):
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
        filename = outpath + str(i) + '.svg'
        plt.savefig(filename)

# Saves raw angles for animations
def plot_kinematics_extract(anglesFile, outpath):
    with open(anglesFile, 'r') as f:
       jsonAngles = json.load(f)

    raw_angles = jsonAngles[0]
    knee_FlexExt = raw_angles['knee_FlexExt']
    hip_FlextExt = raw_angles['hip_FlexExt']
    knee_AbdAdd = raw_angles['knee_AbdAdd']
    hip_AbdAdd = raw_angles['hip_AbdAdd']

    save_angles(knee_FlexExt, 'Knee Flexion/Extension', (-20, 80), outpath + 'knee_FlexExt/')
    save_angles(hip_FlextExt, 'Hip Flexion/Extension', (-20, 60), outpath + 'hip_FlexExt/')
    save_angles(knee_AbdAdd, 'Knee Abduction/Adduction', (-20, 20), outpath + 'knee_AbdAdd/')
    save_angles(hip_AbdAdd, 'Hip Abduction/Adduction', (-30, 30), outpath + 'hip_AbdAdd/')

# Plots kinematics of left or right leg
def plot_angles(angleList, title, yrange, isRed):
    if(isRed): color = red
    else: color = blue
    xmax = len(angleList)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Data points')
    ax.set_ylabel(r"${\Theta}$ (degrees)")
    ax.set(xlim=(0, xmax), ylim=(yrange[0], yrange[1]))
    ax.plot(angleList, color=color)
    plt.show()

# Plots left and right kinematics
def plot_anglesLR(angleList, title, yrange):
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

# Plots each angle list gait cycle in list
def plot_gc(gc, title, yrange, isRed):
    for angleList in gc:
        plot_angles(angleList, title, yrange, isRed)

# Plots left and right gait cycles
def plot_gcLR(gcLR, title, yrange):
    plot_gc(gcLR[0], title, yrange, True)
    plot_gc(gcLR[1], title, yrange, False)

# Plots average as well as standard deviation
def plot_avg(avg, std, title, yrange, N, isRed):
    if (isRed):
        color = red
    else:
        color = blue

    xmax = len(avg)
    fig, ax = plt.subplots()
    ax.set_title(title + ' (' + str(N) + ' Gait Cycles)')
    ax.set_xlabel('Data points')
    ax.set_ylabel(r"${\Theta}$ (degrees)")
    ax.set(xlim=(0, xmax), ylim=(yrange[0], yrange[1]))
    ax.plot(avg, color=color)

    std1_gcL = (np.array(avg) + np.array(std)).tolist()
    std2_gcL = (np.array(avg) - np.array(std)).tolist()
    ax.plot(std1_gcL, '--', color=color)
    ax.plot(std2_gcL, '--', color=color)

# Plots left and right average as well as standard deviation
def plot_avg_gcLR(avg_LR, title, yrange, plotSep):
    avg_gcL = avg_LR['gcL_avg']
    avg_gcR = avg_LR['gcR_avg']
    std_gcL = avg_LR['gcL_std']
    std_gcR = avg_LR['gcR_std']
    N_L = avg_LR['gcL_count']
    N_R = avg_LR['gcR_count']

    if(not plotSep):
        leftMax = len(avg_gcL) - 1
        rightMax = len(avg_gcR) - 1
        xmax = max(leftMax, rightMax)
        fig, ax = plt.subplots()
        ax.set_title(title + ' (' + str(N_L) + 'L, ' + str(N_R) + 'R Gait Cycles)')
        ax.set_xlabel('Data points')
        ax.set_ylabel(r"${\Theta}$ (degrees)")
        ax.set(xlim=(0, xmax), ylim=(yrange[0], yrange[1]))
        ax.plot(avg_gcL, color=red)
        ax.plot(avg_gcR, color=blue)
        plt.show()
    else:
        plot_avg(avg_gcL, std_gcL, title, yrange, N_L, isRed=True)
        plot_avg(avg_gcR, std_gcR, title, yrange, N_R, isRed=False)
        plt.show()

def plot_avg_gcLR_all(gcFile):
    with open(gcFile, 'r') as f:
        gc = json.load(f)

    knee_FlexExt_avg = gc['knee_FlexExt_avg']
    hip_FlexExt_avg = gc['hip_FlexExt_avg']
    knee_AbdAdd_avg = gc['knee_AbdAdd_avg']
    hip_AbdAdd_avg = gc['hip_AbdAdd_avg']

    plot_avg_gcLR(knee_FlexExt_avg, 'Knee Flexion/Extension', (-20, 80), plotSep=False)
    plot_avg_gcLR(hip_FlexExt_avg, 'Hip Flexion/Extension', (-20, 60), plotSep=False)
    plot_avg_gcLR(knee_AbdAdd_avg, 'Knee Abduction/Adduction', (-20, 20), plotSep=False)
    plot_avg_gcLR(hip_AbdAdd_avg, 'Hip Abduction/Adduction', (-30, 30), plotSep=False)

#==================================================================================
#                                   Main
#==================================================================================
path = '..\\Test3\\'
poseFile = path + 'test3.json'
anglesFile = path + 'test3_angles.json'
gcFile = path + 'test3_gc.json'
#plot_avg_gcLR_all(gcFile)

jsonPose_to_plots(poseFile, 'test')