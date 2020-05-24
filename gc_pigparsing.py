#==================================================================================
#                             PLUG IN-GAIT PARSING
#----------------------------------------------------------------------------------
#                       Input: Excel data, Output: JSON
#                  Parses gait cycle excel data processed using
#              Vicon Plug-In-Gait model with marker-based motion capture
#==================================================================================
#                                   Imports
#==================================================================================
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

#==================================================================================
#                                   Methods
#==================================================================================
# Filling in gaps, to cater for ray occlusions in plug in gait
def gapfill(angleList):
    df = pd.DataFrame({'ang': angleList})
    df['ang'].interpolate(method='linear', inplace=True)
    angleList = df['ang'].tolist()
    for i in range(0, len(angleList)):
        if(np.isnan(angleList[i])):
            angleList[i] = angleList[i + 1]
    return angleList

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
#==================================================================================
#                                   Main
#==================================================================================
i = '05'
filePath = '..\\Part'+ i + '\\'
filePIG = filePath + 'Part' + i + '_gc_pig.xlsx'
writeFile = filePath + 'Part' + i + '_gc_pig.json'

# In the same structure as kinematics_processing.py
knee_FlexExt_gc = [[], []]
hip_FlexExt_gc = [[], []]
knee_AbdAdd_gc = [[], []]
hip_AbdAdd_gc = [[], []]

xls = pd.ExcelFile(filePIG)

print('Parsing the PiG sheets...')
for sheet_name in xls.sheet_names:
    df_walk = pd.read_excel(filePIG, sheet_name=sheet_name, header=None)
    df_gc = df_walk.itertuples()

    isDetected = []
    gaitCycleData = []
    for rows in df_gc:
        isDetected.append(rows[4] == 'deg')
        gaitCycleData.append(list(rows[5:106]))

    i = 0
    # Batch parsing gait cycles
    while(True):
        try:
            isDetected[i]
        except IndexError:
            break

        if(isDetected[i]):
            gc = gapfill(gaitCycleData[i])
            hip_FlexExt_gc[1].append(gc)
        if (isDetected[i + 1]):
            gc = gapfill(gaitCycleData[i + 1])
            hip_FlexExt_gc[0].append(gc)
        if (isDetected[i + 2]):
            gc = gapfill(gaitCycleData[i + 2])
            knee_FlexExt_gc[0].append(gc)
        if (isDetected[i + 3]):
            gc = gapfill(gaitCycleData[i + 3])
            knee_FlexExt_gc[1].append(gc)
        if (isDetected[i + 4]):
            gc = gapfill(gaitCycleData[i + 4])
            hip_AbdAdd_gc[0].append(gc)
        if (isDetected[i + 5]):
            gc = gapfill(gaitCycleData[i + 5])
            hip_AbdAdd_gc[1].append(gc)
        if (isDetected[i + 6]):
            gc = gapfill(gaitCycleData[i + 6])
            knee_AbdAdd_gc[0].append(gc)

        # Try catch because if this is the last batch, 7th line can be cut short
        try:
            isDetected[i+7]
        except IndexError:
            break

        if (isDetected[i + 7]):
            gc = gapfill(gaitCycleData[i + 7])
            knee_AbdAdd_gc[1].append(gc)

        i += 9

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

print('Finished!')


