#==================================================================================
#                               FEATURE_EXTRACTION
#----------------------------------------------------------------------------------
#                      Input: Pose sequence, Output: Raw kinematics
#               Given a JSON describing poses throughout two video views,
#               Extracts kinematics and computes kinematics through joint angles
#----------------------------------------------------------------------------------
#==================================================================================
#==================================================================================
#                                   Imports
#==================================================================================
import numpy as np
import matplotlib.pyplot as plt
import json
import time

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
# Calculates joint angle of knee, in terms of flexion and extension
def calc_knee_angle(hip, knee, ankle, rightToLeft):
    # Identifying joint positions
    a = np.array(hip)
    b = np.array(knee)
    c = np.array(ankle)

    # Compute vectors from main joint
    ba = a - b
    m_ba = - ba
    bc = c - b

    cosine_angle = np.dot(m_ba, bc) / (np.linalg.norm(m_ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)

    if(rightToLeft and bc[0] < m_ba[0]): angle = - angle
    if (not rightToLeft and bc[0] > m_ba[0]): angle = - angle
    return angle

# Calculates joint angle of hip, in terms of flexion and extension
def calc_hip_angle(hip, knee, rightToLeft):
    # Identifying joint positions
    a = np.array(hip) # Main joint
    b = np.array(knee)

    # Compute vectors from main joint
    ab = b - a
    m_N = np.array([0,-1])

    cosine_angle = np.dot(ab, m_N) / (np.linalg.norm(ab) * np.linalg.norm(m_N))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)

    if (rightToLeft and ab[0] > m_N[0]): angle = - angle
    if (not rightToLeft and ab[0] < m_N[0]): angle = - angle
    return angle

# Traversing through pose to debug
def side_angles(dataS, dimS, limit=10000):

    knee_FlexExt_L = []
    knee_FlexExt_R = []
    hip_FlexExt_L = []
    hip_FlexExt_R = []

    count = 1
    frames = []

    # Determines whether walking left to right or right to left (essential for joint angles)
    pose_init = dataS[0]
    kneeL_init = pose_init[ptID['knee_L']] # Using knee L as it is the most visible w.r.t gait
    init_x = kneeL_init[0]
    max_x = dimS[0]
    if(init_x > max_x/2): rightToLeft = True
    else: rightToLeft = False

    for pose in dataS:
        frames.append(count)

        #Left
        hip_L = pose[ptID['hip_L']]
        knee_L = pose[ptID['knee_L']]
        ankle_L = pose[ptID['ankle_L']]
        shoulder_L = pose[ptID['shoulder_L']]

        x = [shoulder_L[0], hip_L[0], knee_L[0], ankle_L[0]]
        y = [shoulder_L[1], hip_L[1], knee_L[1], ankle_L[1]]
        angle = calc_knee_angle(hip_L, knee_L, ankle_L, rightToLeft)
        knee_FlexExt_L.append(angle)
        angle = calc_hip_angle(hip_L, knee_L, rightToLeft)
        hip_FlexExt_L.append(angle)

        #Right
        hip_R = pose[ptID['hip_R']]
        knee_R = pose[ptID['knee_R']]
        ankle_R = pose[ptID['ankle_R']]
        shoulder_R = pose[ptID['shoulder_R']]

        x = [shoulder_R[0], hip_R[0], knee_R[0], ankle_R[0]]
        y = [shoulder_R[1], hip_R[1], knee_R[1], ankle_R[1]]
        angle = calc_knee_angle(hip_R, knee_R, ankle_R, rightToLeft)
        knee_FlexExt_R.append(angle)
        angle = calc_hip_angle(hip_R, knee_R, rightToLeft)
        hip_FlexExt_R.append(angle)

        if(count == limit): break
        count += 1

    knee_FlexExt = [knee_FlexExt_L, knee_FlexExt_R]
    hip_FlexExt = [hip_FlexExt_L, hip_FlexExt_R]

    return knee_FlexExt, hip_FlexExt

# Computes and saves kinematics (joint angles) from poses
def calc_angles_jsonPose(jsonFile):
    with open(jsonFile, 'r') as f:
        jsonPose = json.load(f)

    lenF = jsonPose[0]['lenF']
    lenS = jsonPose[0]['lenS']
    #limit = min(lenF, lenS)

    dataS = jsonPose[0]['dataS']
    dimS = jsonPose[0]['dimS']
    knee_FlexExt, hip_FlexExt = side_angles(dataS, dimS)
    jsonDict = {
        'knee_FlexExt' : knee_FlexExt,
        'hip_FlexExt' : hip_FlexExt
    }
    jsonList = [jsonDict]

    #TODO: Decide whether one dic or list of dics (with respect to pose estimation json)
    with open('test_anglesFix' + '.json', 'w') as outfile:
        json.dump(jsonList, outfile, separators=(',', ':'))
        print('dumped! at', outfile)

    return jsonList

jsonFile = 'test.json'
start_time = time.time()
test = calc_angles_jsonPose(jsonFile)
print('JSON raw kinematics file generated:', '{0:.2f}'.format(time.time() - start_time), 's')
