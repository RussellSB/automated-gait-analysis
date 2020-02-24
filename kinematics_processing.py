#==================================================================================
#                               FEATURE_NORMALIZATION
#----------------------------------------------------------------------------------
#                    Input: Pose Json and Raw angles, Output: Gait Cycle graphs
#               Given a JSON describing angles of joints throughout a walk,
#               Smooth kinematics and averages to one standard gait cycle.
#----------------------------------------------------------------------------------
#==================================================================================
#==================================================================================
#                                   Imports
#==================================================================================
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import numpy as np
import json
import math

#==================================================================================
#                                   Constants
#==================================================================================
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
# Filling in gaps, to cater for low confidence in estimation
def gapfill(angleList):
    df = pd.DataFrame({'ang': angleList})
    df['ang'].interpolate(method='linear', inplace=True)
    return df['ang'].tolist()

# Fills gaps of left and right kinematics
def gapfillLR(angLR):
    angL = angLR[0]
    angR = angLR[1]

    filledL = gapfill(angL)
    filledR = gapfill(angR)
    angLR_filled = [filledL, filledR]
    return angLR_filled

# Exponential moving average for a list (naive smoothing)
def smooth1(angle_list, weight):  # Weight between 0 and 1
    last = angle_list[0]  # First value in the plot (first timestep)
    smoothed = []
    for angle in angle_list:
        if(math.isnan(angle) or math.isnan(last)): # Caters for no person detecion
            smoothed.append(None)
            last = angle
        else:
            smoothed_val = last * weight + (1 - weight) * angle  # Calculate smoothed value
            smoothed.append(smoothed_val)
            last = smoothed_val # Anchor the last smoothed value
    return smoothed

def smoothLR(angles_list, weight):
    angles_L = angles_list[0]
    angles_R = angles_list[1]
    smooth_L = smooth1(angles_L, weight)
    smooth_R = smooth1(angles_R, weight)
    smoothed_LR = [smooth_L, smooth_R]

    return smoothed_LR

# Returns list of frames where step on of a particular leg occurs
def getStepOnFrames(dataS, L_or_R, hist, avg_thresh):
    ankle_points = []
    isGrounded_srs = []
    stepOnFrames = []

    seekStepOn = True

    for i in range(0, len(dataS)):
        pose = dataS[i]
        isGrounded = False

        ankle_pos = pose[ptID['ankle_' + L_or_R]]
        ankle_X = ankle_pos[0]
        ankle_Y = ankle_pos[1]

        if (i > 0 and (ankle_pos != [-1,-1] or ankle_points[-1] != [-1,-1]) ):
            ankle_pos_prev = ankle_points[-1]
            ankle_X_prev = ankle_pos_prev[0]
            ankle_Y_prev = ankle_pos_prev[1]

            X_diff = pow(abs(ankle_X - ankle_X_prev), 2)
            Y_diff = pow(abs(ankle_Y - ankle_Y_prev), 1)

            abs_diff = Y_diff + X_diff

            if (abs_diff < 5): isGrounded = True

            isGrounded_recent = isGrounded_srs[-hist:]
            isGrounded_avg = sum(isGrounded_recent)/len(isGrounded_recent)

            # print(i, ankle_pos, abs_diff, isGrounded, isGrounded_avg)

            if(seekStepOn):
                if(isGrounded_avg > avg_thresh):
                    stepOnFrames.append(i-hist)
                    seekStepOn = False
            else:
                if(isGrounded_avg == 0):
                    seekStepOn = True
        ankle_points.append(pose[ptID['ankle_' + L_or_R]])
        isGrounded_srs.append(isGrounded)
    return stepOnFrames

# Returns set of subsets for gait cycles
def gaitCycle_filter(angle_list, stepOnFrames):
    gc = [] # gait cycle list to store subsets
    for i in range(len(stepOnFrames) - 1, 0, -1):
        end = stepOnFrames[i] - 1
        start = stepOnFrames[i-1]

        if(start >= 0):
            subset = angle_list[start:end]
            gc.append(subset)
    return gc

# Returns right and left gait cycles of angle list
def gcLR(angleList, stepOnFrames_L, stepOnFrames_R):
    gc_L = gaitCycle_filter(angleList[0], stepOnFrames_L)
    gc_R = gaitCycle_filter(angleList[1], stepOnFrames_R)
    gc = [gc_L, gc_R]
    return gc

# TODO: Improve catering for None
# Normalizes the xrange to a sample of N data points
def resample_gcLR(gcLR, N):
    gcL = gcLR[0]
    gcR = gcLR[1]
    gcLR_resampled = [[], []]

    for angleList in gcL:

        for i in range(0,len(angleList)):
            if(angleList[i] == None):
                angleList[i] = 0
        angleListL = signal.resample(angleList, N)
        gcLR_resampled[0].append(angleListL)

    for angleList in gcR:
        for i in range(0,len(angleList)):
            if(angleList[i] == None):
                angleList[i] = 0
        angleListR = signal.resample(angleList, N)
        gcLR_resampled[1].append(angleListR)

    return gcLR_resampled

# Returns average of left and right gait cycles respectively
def avg_gcLR(gcLR):
    gcL = np.array(gcLR[0]) # list of left gait cycles
    gcR = np.array(gcLR[1]) # list of right gait cycles

    gcL_avg = np.mean(gcL, axis=0)
    gcL_std = np.std(gcL, axis=0)

    gcR_avg = np.mean(gcR, axis=0)
    gcR_std = np.std(gcR, axis=0)

    avg_gcLR = {
        'gcL_avg' : gcL_avg.tolist(),
        'gcL_std' : gcL_std.tolist(),
        'gcR_avg': gcR_avg.tolist(),
        'gcR_std': gcR_std.tolist(),
        'gcL_count' : len(gcL),
        'gcR_count' : len(gcR)
    }
    return avg_gcLR

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
        leftMax = len(avg_gcL)
        rightMax = len(avg_gcR)
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

#==================================================================================
#                                   Main
#==================================================================================
with open('../Test/test.json', 'r') as f:
    jsonPose = json.load(f)

dataS = jsonPose[0]['dataS']
dimS = jsonPose[0]['dimS']
dataF = jsonPose[0]['dataF']
dimF = jsonPose[0]['dimF']
lenS = jsonPose[0]['lenS']

with open('../Test/test_angles.json', 'r') as f:
    jsonAngles = json.load(f)
raw_angles = jsonAngles[0]

knee_FlexExt = raw_angles['knee_FlexExt']
hip_FlexExt = raw_angles['hip_FlexExt']
knee_AbdAdd = raw_angles['knee_AbdAdd']
hip_AbdAdd = raw_angles['hip_AbdAdd']

# Gap filling
knee_FlexExt0 = gapfillLR(knee_FlexExt)
hip_FlexExt0 = gapfillLR(hip_FlexExt)
knee_AbdAdd0 = gapfillLR(knee_AbdAdd)
hip_AbdAdd0 = gapfillLR(hip_AbdAdd)

# Smoothing
weight = 0.8
knee_FlexExt1 = smoothLR(knee_FlexExt0, weight)
hip_FlexExt1 = smoothLR(hip_FlexExt0, weight)
knee_AbdAdd1 = smoothLR(knee_AbdAdd0, weight)
hip_AbdAdd1 = smoothLR(hip_AbdAdd0, weight)

# Slicing into gait cycles
stepOnFrames_L = getStepOnFrames(dataS, 'L',  6, 0.6)
stepOnFrames_R = getStepOnFrames(dataS, 'R',  6, 0.6)
knee_FlexExt2 = gcLR(knee_FlexExt1, stepOnFrames_L, stepOnFrames_R)
hip_FlexExt2 = gcLR(hip_FlexExt1, stepOnFrames_L, stepOnFrames_R)
knee_AbdAdd2 = gcLR(knee_AbdAdd1, stepOnFrames_L, stepOnFrames_R)
hip_AbdAdd2 = gcLR(hip_AbdAdd1, stepOnFrames_L, stepOnFrames_R)

# Resampling to 100 (100 and 0 inclusive)
knee_FlexExt3 = resample_gcLR(knee_FlexExt2, 101)
hip_FlexExt3 = resample_gcLR(hip_FlexExt2, 101)
knee_AbdAdd3 = resample_gcLR(knee_AbdAdd2, 101)
hip_AbdAdd3 = resample_gcLR(hip_AbdAdd2, 101)

# Averaging
knee_FlexExt4 = avg_gcLR(knee_FlexExt3)
hip_FlexExt4 = avg_gcLR(hip_FlexExt3)
knee_AbdAdd4 = avg_gcLR(knee_AbdAdd3)
hip_AbdAdd4 = avg_gcLR(hip_AbdAdd3)

#plot_anglesLR(knee_FlexExt, 'Knee Flexion/Extension', (-20, 80)) # Orig
#plot_anglesLR(knee_FlexExt1, 'Knee Flexion/Extension', (-20, 80)) # Gap Fill + smoothing
#plot_gcLR(knee_FlexExt3, 'Knee Flexion/Extension', (-20, 80)) # Gait cycle splitting + resampling
#plot_avg_gcLR(knee_FlexExt4, 'Knee Flexion/Extension', (-20, 80), plotSep=False) # Avg and std
# plot_avg_gcLR(knee_FlexExt4, 'Knee Flexion/Extension', (-20, 80), plotSep=True) # Avg and std

plot_avg_gcLR(knee_FlexExt4, 'Knee Flexion/Extension', (-20, 80), plotSep=True) # Avg and std
plot_avg_gcLR(hip_FlexExt4, 'Hip Flexion/Extension', (-20, 60), plotSep=True) # Avg and std
plot_avg_gcLR(knee_AbdAdd4, 'Knee Abduction/Adduction', (-20, 20), plotSep=True) # Avg and std
plot_avg_gcLR(hip_AbdAdd4, 'Hip Abduction/Adduction', (-30, 30), plotSep=True) # Avg and std

#plot_anglesLR(hip_FlexExt, 'Hip Flexion/Extension', (-20, 60)) # Orig
#plot_anglesLR(hip_FlexExt1, 'Hip Flexion/Extension', (-20, 60)) # Gap Fill
#plot_gcLR(hip_FlexExt3, 'Hip Flexion/Extension', (-20, 60)) # Gait cycle splitting + Smoothing
#plot_avg_gcLR(hip_FlexExt4, 'Hip Flexion/Extension', (-20, 60), plotSep=False) # Avg and std
#plot_avg_gcLR(hip_FlexExt4, 'Hip Flexion/Extension', (-20, 60), plotSep=True) # Avg and std

# plot_anglesLR(knee_AbdAdd, 'Knee Abduction/Adduction', (-20, 20))
# plot_anglesLR(knee_AbdAdd1, 'Knee Abduction/Adduction', (-20, 20))
# plot_gcLR(knee_AbdAdd3, 'Knee Abduction/Adduction', (-20, 20))

# plot_anglesLR(knee_FlexExt, 'Knee Flexion/Extension', (-20, 80))
# plot_anglesLR(hip_FlexExt, 'Hip Flexion/Extension', (-20, 60))
# plot_anglesLR(knee_AbdAdd, 'Knee Abduction/Adduction', (-20, 20))
# plot_anglesLR(hip_AbdAdd, 'Hip Abduction/Adduction', (-30, 30))