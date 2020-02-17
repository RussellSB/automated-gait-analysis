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

#==================================================================================
#                                   Methods
#==================================================================================
# Calculates angle of joint b
def calc_angle(threeJoints):
    # Identifying joint positions
    a = np.array(threeJoints[0])
    b = np.array(threeJoints[1]) # Main joint
    c = np.array(threeJoints[2])

    # Compute vectors from main joint
    ba = a - b
    m_ba = - ba
    bc = c - b

    cosine_angle = np.dot(m_ba, bc) / (np.linalg.norm(m_ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Traversing through pose to debug
def kinematics_raw(data, dim, limit):
    count = 1
    frames, anglesL, anglesR = [], [], []

    for pose in data:
        fig, axes = plt.subplots(2, 1)
        axes[0].set_title('Knee Flex/Extension')
        axes[0].set(xlim=(0, dim[0]), ylim=(0, dim[1]))  # setting width and height of plot
        axes[1].set(xlim=(0, len(data)), ylim=(-20, 80))  # setting width and height of plot
        axes[1].set_xlabel('Frame (count)')
        axes[1].set_ylabel(r"$\dot{\Theta}$ (degrees)")

        kneeL = []
        kneeR = []
        xL, yL = [], []
        xR, yR = [], []

        for i in range(0, len(pose)):
            if(i == 11): # HipL
                xL.append(pose[i][0])
                yL.append(pose[i][1])
                kneeL.append(pose[i])
            if (i == 13):  # KneeL
                xL.append(pose[i][0])
                yL.append(pose[i][1])
                kneeL.append(pose[i])
            if (i == 15):  # AnkleL
                xL.append(pose[i][0])
                yL.append(pose[i][1])
                kneeL.append(pose[i])
                angleL = calc_angle(kneeL)

                anglesL.append(angleL)

            if (i == 12):  # HipR
                xR.append(pose[i][0])
                yR.append(pose[i][1])
                kneeR.append(pose[i])
            if (i == 14):  # KneeR
                xR.append(pose[i][0])
                yR.append(pose[i][1])
                kneeR.append(pose[i])
            if (i == 16):  # AnkleR
                xR.append(pose[i][0])
                yR.append(pose[i][1])
                kneeR.append(pose[i])
                angleR = calc_angle(kneeR)

                frames.append(count)
                anglesR.append(angleR)

                blue = "#72B6E9"
                axes[0].plot(xR, yR, color=blue)
                axes[0].scatter(xR, yR, s=20, color=blue)
                axes[1].plot(frames, anglesR, blue)

                red = "#FF4A7E"
                axes[0].plot(xL, yL, color=red)
                axes[0].scatter(xL, yL, s=20, color=red)
                axes[1].plot(frames, anglesL, red)

                plt.savefig('../TEST/GIF/knee-F1/' + str(count) + '.svg')
        if(count == limit): break
        count += 1

    return anglesL, anglesR

# Makes a collection of figures out of what is described in the jsonFile
def jsonPose_to_pics(jsonFile, path):
    with open('test.json', 'r') as f:
        jsonPose = json.load(f)

    lenF = jsonPose[0]['lenF']
    lenS = jsonPose[0]['lenS']
    # limit = min(lenF, lenS)
    limit = 1000

    dataS = jsonPose[0]['dataS']
    dimS = jsonPose[0]['dimS']
    anglesL, anglesR = kinematics_raw(dataS, dimS, limit)

    #dataF = jsonPose[0]['dataF']
    #dimF = jsonPose[0]['dimF']
    #kinematics_raw(dataF, dimF, limit)

path = '../Test/GIF/'
jsonPose_to_pics('test.json', path)