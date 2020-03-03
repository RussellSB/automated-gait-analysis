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

#==================================================================================
#                                   Constants
#==================================================================================
# Normal/Abnormal, Age, Gender
partInfo = {
    '1':('N', 23, 'M'), '2':('N', 20, 'M'), '3':('N', 20, 'M'),
    '4':('N', 20, 'F'), '5':('N', 22, 'M'), '6':('N', 19, 'M'),
    '7':('N', 20, 'F'), '8':('N', 20, 'M'), '9':('N', 22, 'F'),
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

#==================================================================================
#                                   Main
#==================================================================================
data = []
labels_id = []
labels_na = []
labels_age = []
labels_gen = []

for i in range(1, 10):
    id = str(i)
    id = '0' + id if len(id) < 2 else i
    part = 'Part' + str(id)

    file = '..\\' + part + '\\' + part + '_gc.json'
    with open(file, 'r') as f:
        gc_PE = json.load(f)

    kinematics = getgc_glob(gc_PE)
    for gc in kinematics:
        na = partInfo[str(i)][0]
        age = partInfo[str(i)][1]
        gen = partInfo[str(i)][2]

        data.append(gc)
        labels_id.append(i)
        labels_na.append(na)
        labels_age.append(age)
        labels_gen.append(gen)

    with open('..\\data.pickle', 'wb') as f:
        pickle.dump(data, f)
    with open('..\\labels_id.pickle', 'wb') as f:
        pickle.dump(labels_id, f)
    with open('..\\labels_na.pickle', 'wb') as f:
        pickle.dump(labels_na, f)
    with open('..\\labels_age.pickle', 'wb') as f:
        pickle.dump(labels_age, f)
    with open('..\\labels_gen.pickle', 'wb') as f:
        pickle.dump(labels_gen, f)