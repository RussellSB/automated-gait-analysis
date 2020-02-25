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
import time

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
        gcLR_resampled[0].append(angleListL.tolist())

    for angleList in gcR:
        for i in range(0,len(angleList)):
            if(angleList[i] == None):
                angleList[i] = 0
        angleListR = signal.resample(angleList, N)
        gcLR_resampled[1].append(angleListR.tolist())

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

def kinematics_process(poseFile, anglesFile, writeFile):
    with open(poseFile, 'r') as f:
        jsonPose = json.load(f)
    with open(anglesFile, 'r') as f:
        jsonAngles = json.load(f)

    len1 = len(jsonPose)
    len2 = len(jsonAngles)
    if (len1 != len2):
        print('Error: jsonPose of len', len1, 'does not match jsonAngles of len', len2)
        exit()

    knee_FlexExt_gc = [[], []]
    hip_FlexExt_gc = [[], []]
    knee_AbdAdd_gc = [[], []]
    hip_AbdAdd_gc = [[], []]

    for i in range(0, len1):
        pose_srs = jsonPose[i]

        dataS = pose_srs['dataS']
        dimS = pose_srs['dimS']
        dataF = pose_srs['dataF']
        dimF = pose_srs['dimF']
        lenS = pose_srs['lenS']

        raw_angles = jsonAngles[i]

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
        stepOnFrames_L = getStepOnFrames(dataS, 'L', 8, 0.8)
        stepOnFrames_R = getStepOnFrames(dataS, 'R', 8, 0.8)
        knee_FlexExt2 = gcLR(knee_FlexExt1, stepOnFrames_L, stepOnFrames_R)
        hip_FlexExt2 = gcLR(hip_FlexExt1, stepOnFrames_L, stepOnFrames_R)
        knee_AbdAdd2 = gcLR(knee_AbdAdd1, stepOnFrames_L, stepOnFrames_R)
        hip_AbdAdd2 = gcLR(hip_AbdAdd1, stepOnFrames_L, stepOnFrames_R)

        # Resampling to 100 (100 and 0 inclusive)
        knee_FlexExt3 = resample_gcLR(knee_FlexExt2, 101)
        hip_FlexExt3 = resample_gcLR(hip_FlexExt2, 101)
        knee_AbdAdd3 = resample_gcLR(knee_AbdAdd2, 101)
        hip_AbdAdd3 = resample_gcLR(hip_AbdAdd2, 101)

        # Adding to global gait cycle instances list
        for gc in knee_FlexExt3[0]: knee_FlexExt_gc[0].append(gc)
        for gc in knee_FlexExt3[1]: knee_FlexExt_gc[1].append(gc)

        for gc in hip_FlexExt3[0]: hip_FlexExt_gc[0].append(gc)
        for gc in hip_FlexExt3[1]: hip_FlexExt_gc[1].append(gc)

        for gc in knee_AbdAdd3[0]: knee_AbdAdd_gc[0].append(gc)
        for gc in knee_AbdAdd3[1]: knee_AbdAdd_gc[1].append(gc)

        for gc in hip_AbdAdd3[0]: hip_AbdAdd_gc[0].append(gc)
        for gc in hip_AbdAdd3[1]: hip_AbdAdd_gc[1].append(gc)

    # Averaging
    knee_FlexExt_avg = avg_gcLR(knee_FlexExt_gc)
    hip_FlexExt_avg = avg_gcLR(hip_FlexExt_gc)
    knee_AbdAdd_avg = avg_gcLR(knee_AbdAdd_gc)
    hip_AbdAdd_avg = avg_gcLR(hip_AbdAdd_gc)

    jsonDict = {
        'knee_FlexExt_avg': knee_FlexExt_avg,
        'hip_FlexExt_avg': hip_FlexExt_avg,
        'knee_AbdAdd_avg': knee_AbdAdd_avg,
        'hip_AbdAdd_avg': hip_AbdAdd_avg,

        'knee_FlexExt_gc': knee_FlexExt_gc,
        'hip_FlexExt_gc': hip_FlexExt_gc,
        'knee_AbdAdd_gc': knee_AbdAdd_gc,
        'hip_AbdAdd_gc': hip_AbdAdd_gc,
    }

    with open(writeFile, 'w') as outfile:
        json.dump(jsonDict, outfile, separators=(',', ':'))

#==================================================================================
#                                   Main
#==================================================================================
path = '..\\Test3\\'
poseFile = path + 'test3.json'
anglesFile = path + 'test3_angles.json'
writeFile = path + 'test3_gc.json'
start_time = time.time()
kinematics_process(poseFile, anglesFile, writeFile)
print('Kinematics processed and saved in', '\"'+writeFile+'\"', '[Time:', '{0:.2f}'.format(time.time() - start_time), 's]')