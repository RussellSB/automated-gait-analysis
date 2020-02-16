#==================================================================================
#                               FEATURE_EXTRACTION
#----------------------------------------------------------------------------------
#                           Input: JSON, Output: Gait Cycle graphs
#               Given a JSON describing poses throughout two video views,
#               Extracts kinematics and computes kinematic graphs through angles
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
def plot_debug(data, dim, limit):
    count = 1
    fig, ax = plt.subplots()
    frames, angles = [], []

    for pose in data:
        #print(count)

        #fig, ax = plt.subplots()
        #ax.set(xlim=(0, dim[0]), ylim=(0, dim[1]))  # setting width and height of plot

        leftHip = []
        x, y = [], []

        for i in range(0, len(pose)):
            if(i == 11): # Hip
                x.append(pose[i][0])
                y.append(pose[i][1])
                leftHip.append(pose[i])
            if (i == 13):  # Knee
                x.append(pose[i][0])
                y.append(pose[i][1])
                leftHip.append(pose[i])
            if (i == 15):  # Ankle
                x.append(pose[i][0])
                y.append(pose[i][1])
                leftHip.append(pose[i])
                angle = calc_angle(leftHip)
                print(count, angle)

                frames.append(count)
                angles.append(angle)
        #ax.scatter(x, y, s=20)
        #plt.show()
        if(count == limit): break
        count += 1

    ax.plot(frames, angles)
    plt.show()
    print('fin')

# Makes a collection of figures out of what is described in the jsonFile
def jsonPose_to_pics(jsonFile, path):
    with open('test.json', 'r') as f:
        jsonPose = json.load(f)

    lenF = jsonPose[0]['lenF']
    lenS = jsonPose[0]['lenS']
    # limit = min(lenF, lenS)
    limit = 350

    dataS = jsonPose[0]['dataS']
    dimS = jsonPose[0]['dimS']
    plot_debug(dataS, dimS, limit)

    #dataF = jsonPose[0]['dataF']
    #dimF = jsonPose[0]['dimF']
    #calc_angle(dataF, dimF, limit)

path = '../Test/GIF/'
jsonPose_to_pics('test.json', path)