#==================================================================================
#                               DATA PREPROCESSING
#----------------------------------------------------------------------------------
#                      Input: Gait cycles, Output: Datasets
#               Prepares dataset from the kinematics, in a format
#               ready for classification of gait cycles.
#==================================================================================
#                                   Imports
#==================================================================================
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import random
import math

#==================================================================================
#                                   Constants
#==================================================================================
# Normality, Age, Gender
partInfo = {
    # Normal participants
    '1':('N', 23, 'M'), '2':('N', 20, 'M'), '3':('N', 20, 'M'),
    '4':('N', 20, 'F'), '5':('N', 22, 'M'), '6':('N', 19, 'M'),
    '7':('N', 20, 'F'), '8':('N', 20, 'M'), '9':('N', 22, 'F'),
    '10':('N', 22, 'F'), '11':('N', 19, 'F'), '12':('N', 20, 'M'),
    '13':('N', 20, 'F'), '14':('N', 20, 'M'), '15':('N', 78, 'F'),
    '16':('N', 80, 'M'), '17':('N', 20, 'F'),
    # Abnormal participants (simulations from Part14)
    '18':('A', 20, 'M'), '19':('A', 20, 'M'), '20':('A', 20, 'M'),
    '21':('A', 20, 'M')
}

#==================================================================================
#                                   Methods
#==================================================================================
# Returns list of 2d arrays of all of the participant's gait cycles
def getgc_glob(gc_PE):
    # Left
    knee_FlexExt_L = gc_PE['knee_FlexExt_gc'][0]
    hip_FlexExt_L = gc_PE['hip_FlexExt_gc'][0]
    knee_AbdAdd_L = gc_PE['knee_AbdAdd_gc'][0]
    hip_AbdAdd_L = gc_PE['hip_AbdAdd_gc'][0]
    len_gcL = len(knee_FlexExt_L)

    # Right
    knee_FlexExt_R = gc_PE['knee_FlexExt_gc'][1]
    hip_FlexExt_R = gc_PE['hip_FlexExt_gc'][1]
    knee_AbdAdd_R = gc_PE['knee_AbdAdd_gc'][1]
    hip_AbdAdd_R = gc_PE['hip_AbdAdd_gc'][1]
    len_gcR = len(knee_FlexExt_R)

    len_min = min(len_gcL, len_gcR) # must set a minumum for matching and consistent L and R gait cycles

    # Reversing order of gait cycles so that they are consistent with each other when normalized
    knee_FlexExt_L.reverse()
    knee_AbdAdd_L.reverse()
    hip_FlexExt_L.reverse()
    hip_AbdAdd_L.reverse()
    knee_FlexExt_R.reverse()
    hip_FlexExt_R.reverse()
    knee_AbdAdd_R.reverse()
    hip_AbdAdd_R.reverse()

    kinematics = []
    for i in range(0, len_min):
        arr2d = []

        arr2d.append(knee_FlexExt_L[i])
        arr2d.append(knee_FlexExt_R[i])
        arr2d.append(hip_FlexExt_L[i])
        arr2d.append(hip_FlexExt_R[i])

        arr2d.append(knee_AbdAdd_L[i])
        arr2d.append(knee_AbdAdd_R[i])
        arr2d.append(hip_AbdAdd_L[i])
        arr2d.append(hip_AbdAdd_R[i])

        arr2d = np.array(arr2d)
        kinematics.append(arr2d)
    print(len(kinematics))
    return kinematics

# Returns a list of n 2d arrays that artificially simulate abnormal gait cycles
def get_gcart(n):
    kinematics_artificial = []

    for _ in range(0, n):
        arr2d = []

        k = [1 / 4, 1 / 2, 1, 2]  # Constants for xrange
        x = np.linspace(0, 12.5 * random.choice(k), 101)
        noise = np.random.normal(0, 0.1, 101)

        knee_FlexExt = np.sin(x) * 30 + noise + 30
        hip_FlexExt = np.cos(x) * 30 + noise + 10
        knee_AbdAdd = np.sin(x) * 5 + noise + 5
        hip_AbdAdd = np.sin(x) * 10 + noise

        for _ in range(0,2): arr2d.append(knee_FlexExt)
        for _ in range(0, 2): arr2d.append(hip_FlexExt)
        for _ in range(0, 2): arr2d.append(knee_AbdAdd)
        for _ in range(0, 2): arr2d.append(hip_AbdAdd)

        arr2d = np.array(arr2d)
        kinematics_artificial.append(arr2d)
    return kinematics_artificial

#==================================================================================
#                                   Main
#==================================================================================
data = []
labels_id = []
labels_age = []
labels_gen = []

data_na = []
labels_na = []
labels_id_na = []

# Prepares gait data collected from the lab
for i in range(1, 22):
    id = str(i)
    id = '0' + id if len(id) < 2 else i
    part = 'Part' + str(id)

    file = '..\\' + part + '\\' + part + '_gc.json'
    with open(file, 'r') as f:
        gc_PE = json.load(f)

    kinematics = getgc_glob(gc_PE)

    for gc in kinematics:
        na = partInfo[str(i)][0]
        #na = 0 if na == 'N' else 1
        na = 'Normal' if na == 'N' else 'Abnormal'

        gen = partInfo[str(i)][2]
        #gen = 0 if gen == 'F' else 1
        gen = 'Female' if gen == 'F' else 'Male'

        age = partInfo[str(i)][1]

        # Separate abnormal/normal set from normal set
        if(na=='Abnormal'):
            data_na.append(gc)
            labels_na.append(na)
            labels_id_na.append(i)
        else:
            data.append(gc)
            labels_id.append(i)
            labels_age.append(age)
            labels_gen.append(gen)

            data_na.append(gc)
            labels_na.append(na)
            labels_id_na.append(i)

# Prepares artificial gait data simulating abnormalities
#kinematics_artificial = get_gcart(int(len(data)/2))
#data_na = [x for x in data]
#for gc in kinematics_artificial:
#    data_na.append(gc)
#    labels_na.append('Abnormal')

with open('..\\classifier_data\\data.pickle', 'wb') as f:
    pickle.dump(data, f)
with open('..\\classifier_data\\labels_id.pickle', 'wb') as f:
    pickle.dump(labels_id, f)
with open('..\\classifier_data\\labels_age.pickle', 'wb') as f:
    pickle.dump(labels_age, f)
with open('..\\classifier_data\\labels_gender.pickle', 'wb') as f:
    pickle.dump(labels_gen, f)

with open('..\\classifier_data\\data_na.pickle', 'wb') as f:
    pickle.dump(data_na, f)
with open('..\\classifier_data\\labels_id_na.pickle', 'wb') as f:
    pickle.dump(labels_id_na, f)
with open('..\\classifier_data\\labels_abnormality.pickle', 'wb') as f:
    pickle.dump(labels_na, f)