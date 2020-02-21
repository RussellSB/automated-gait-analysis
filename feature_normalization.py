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
def plot_angles(angleList, title, yrange, isRed):
    if(isRed): color = "#FF4A7E"
    else: color = "#72B6E9"
    xmax = len(angleList)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Frame (count)')
    ax.set_ylabel(r"${\Theta}$ (degrees)")
    ax.set(xlim=(0, xmax), ylim=(yrange[0], yrange[1]))
    ax.plot(angleList, color=color)
    plt.show()

# Plots left and right kinematics
def plot_anglesLR(angleList, title, yrange):
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

# Plots each angle list gait cycle in list
def plot_gc(gc, title, yrange, isRed):
    for angleList in gc:
        plot_angles(angleList, title, yrange, isRed)

# Plots left and right gait cycles
def plot_gcLR(gcLR, title, yrange):
    plot_gc(gcLR[0], title, yrange, True)
    plot_gc(gcLR[1], title, yrange, False)

# TODO: Test that it works for other participants too
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

        if (i > 0):
            ankle_pos_prev = ankle_points[-1]
            ankle_X_prev = ankle_pos_prev[0]
            ankle_Y_prev = ankle_pos_prev[1]

            X_diff = pow(abs(ankle_X - ankle_X_prev), 2)
            Y_diff = pow(abs(ankle_Y - ankle_Y_prev), 1)

            abs_diff = Y_diff + X_diff

            if (abs_diff < 5): isGrounded = True

            isGrounded_recent = isGrounded_srs[-hist:]
            isGrounded_avg = sum(isGrounded_recent)/len(isGrounded_recent)

            #print(i, abs_diff, isGrounded, isGrounded_avg)

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

# Exponential moving average for a list (naive smoothing)
def smooth(angle_list, weight):  # Weight between 0 and 1
    last = angle_list[0]  # First value in the plot (first timestep)
    smoothed = []
    for angle in angle_list:
        smoothed_val = last * weight + (1 - weight) * angle  # Calculate smoothed value
        smoothed.append(smoothed_val) # Save it
        last = smoothed_val # Anchor the last smoothed value
    return smoothed

#TODO: Try out other smoothing methods, but rn focus on gait cycle extraction
def smoothLR(angles_list, weight):
    angles_L = angles_list[0]
    angles_R = angles_list[1]
    smooth_L = smooth(angles_L, weight)
    smooth_R = smooth(angles_R, weight)
    smoothed_LR = [smooth_L, smooth_R]

    return smoothed_LR

#==================================================================================
#                                   Main
#==================================================================================

with open('test.json', 'r') as f:
    jsonPose = json.load(f)

dataS = jsonPose[0]['dataS']
dimS = jsonPose[0]['dimS']
dataF = jsonPose[0]['dataF']
dimF = jsonPose[0]['dimF']
lenS = jsonPose[0]['lenS']

with open('test_angles3.json', 'r') as f:
    jsonAngles = json.load(f)
raw_angles = jsonAngles[0]

knee_FlexExt = raw_angles['knee_FlexExt']
hip_FlexExt = raw_angles['hip_FlexExt']
knee_AbdAdd = raw_angles['knee_AbdAdd']
hip_AbdAdd = raw_angles['hip_AbdAdd']

# Smoothing
weight = 0.8
knee_FlexExt1 = smoothLR(knee_FlexExt, weight)
hip_FlexExt1 = smoothLR(hip_FlexExt, weight)
knee_AbdAdd1 = smoothLR(knee_AbdAdd, weight)
hip_AbdAdd1 = smoothLR(hip_AbdAdd, weight)

# Slicing into gait cycles
stepOnFrames_L = getStepOnFrames(dataS, 'L',  6, 0.6)
stepOnFrames_R = getStepOnFrames(dataS, 'R',  6, 0.6)
knee_FlexExt2 = gcLR(knee_FlexExt1, stepOnFrames_L, stepOnFrames_R)
hip_FlexExt2 = gcLR(hip_FlexExt1, stepOnFrames_L, stepOnFrames_R)
knee_AbdAdd2 = gcLR(knee_AbdAdd, stepOnFrames_L, stepOnFrames_R)
hip_AbdAdd2 = gcLR(hip_AbdAdd1, stepOnFrames_L, stepOnFrames_R)

plot_anglesLR(knee_FlexExt, 'Knee Flexion/Extension', (-20, 80))
plot_anglesLR(knee_FlexExt1, 'Knee Flexion/Extension', (-20, 80))
plot_gcLR(knee_FlexExt2, 'Knee Flexion/Extension', (-20, 80))

plot_anglesLR(knee_AbdAdd, 'Knee Abduction/Adduction', (-20, 20))
plot_anglesLR(knee_AbdAdd1, 'Knee Abduction/Adduction', (-20, 20))
plot_gcLR(knee_AbdAdd2, 'Knee Abduction/Adduction', (-20, 20))


# plot_anglesLR(knee_FlexExt, 'Knee Flexion/Extension', (-20, 80))
# plot_anglesLR(hip_FlexExt, 'Hip Flexion/Extension', (-20, 60))
# plot_anglesLR(knee_AbdAdd, 'Knee Abduction/Adduction', (-20, 20))
# plot_anglesLR(hip_AbdAdd, 'Hip Abduction/Adduction', (-30, 30))



