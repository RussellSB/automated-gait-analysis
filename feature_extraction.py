#==================================================================================
#                               FEATURE_EXTRACTION
#----------------------------------------------------------------------------------
#                           Input: JSON, Output: Gait Cycle graphs
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
def calc_knee_angle(joint1, joint2, joint3):
    # Identifying joint positions
    a = np.array(joint1)
    b = np.array(joint2) # Main joint
    c = np.array(joint3)

    # Compute vectors from main joint
    ba = a - b
    m_ba = - ba
    bc = c - b

    cosine_angle = np.dot(m_ba, bc) / (np.linalg.norm(m_ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Calculates joint angle of hip, in terms of flexion and extension
def calc_hip_angle(hip, knee):
    # Identifying joint positions
    a = np.array(hip) # Main joint
    b = np.array(knee)

    # Compute vectors from main joint
    ab = b - a
    m_N = np.array([0,-1])

    cosine_angle = np.dot(ab, m_N) / (np.linalg.norm(ab) * np.linalg.norm(m_N))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Traversing through pose to debug
def side_angles(dataS, dim, limit=1000):

    knee_FlexExt_L = []
    knee_FlexExt_R = []
    hip_FlexExt_L = []
    hip_FlexExt_R = []

    red = "#FF4A7E"
    blue = "#72B6E9"

    count = 1
    frames = []

    fig, ax = plt.subplots(2, 1)
    ax[0].set_title('Hip Flex / Extension')
    ax[1].set_xlabel('Frame (count)')
    ax[1].set_ylabel(r"${\Theta}$ (degrees)")
    ax[0].set(xlim=(0, dim[0]), ylim=(0, dim[1]))
    ax[1].set(xlim=(0, len(dataS)), ylim=(-20, 60))

    for pose in dataS:
        frames.append(count)

        #Left
        hip_L = pose[ptID['hip_L']]
        knee_L = pose[ptID['knee_L']]
        ankle_L = pose[ptID['ankle_L']]
        shoulder_L = pose[ptID['shoulder_L']]
        #if(hip_L == [-1,-1] or knee_L == [-1,-1] or ankle_L == [-1,-1] or shoulder_L == [-1, -1]): break
        
        x = [shoulder_L[0], hip_L[0], knee_L[0], ankle_L[0]]
        y = [shoulder_L[1], hip_L[1], knee_L[1], ankle_L[1]]
        angle = calc_knee_angle(hip_L, knee_L, ankle_L)
        knee_FlexExt_L.append(angle)
        angle = calc_hip_angle(hip_L, knee_L)
        hip_FlexExt_L.append(angle)

        ax[0].scatter(x, y, s=20, color=red)
        ax[0].plot(x, y, color=red)
        ax[1].plot(frames, hip_FlexExt_L, color=red)

        #Right
        hip_R = pose[ptID['hip_R']]
        knee_R = pose[ptID['knee_R']]
        ankle_R = pose[ptID['ankle_R']]
        shoulder_R = pose[ptID['shoulder_R']]
        #if(hip_R == [-1,-1] or knee_R == [-1,-1] or ankle_R == [-1,-1] or shoulder_R == [-1, -1]): break

        x = [shoulder_R[0], hip_R[0], knee_R[0], ankle_R[0]]
        y = [shoulder_R[1], hip_R[1], knee_R[1], ankle_R[1]]
        angle = calc_knee_angle(hip_R, knee_R, ankle_R)
        knee_FlexExt_R.append(angle)
        angle = calc_hip_angle(hip_R, knee_R)
        hip_FlexExt_R.append(angle)

        ax[0].scatter(x, y, s=20, color=blue)
        ax[0].plot(x, y, color=blue)
        ax[1].plot(frames, hip_FlexExt_R, color=blue)

        if(count == limit): break
        count += 1

    plt.show()
    knee_FlexExt = [knee_FlexExt_L, knee_FlexExt_R]
    #return knee_FlexExt

# Makes a collection of figures out of what is described in the jsonFile
def parse_jsonPose(jsonFile):
    with open('test.json', 'r') as f:
        jsonPose = json.load(f)

    lenF = jsonPose[0]['lenF']
    lenS = jsonPose[0]['lenS']
    limit = min(lenF, lenS)

    dataS = jsonPose[0]['dataS']
    dimS = jsonPose[0]['dimS']
    side_angles(dataS, dimS, limit)
