#==================================================================================
#                               KINEMATICS_EXTRACTION
#----------------------------------------------------------------------------------
#                      Input: Pose sequence, Output: Raw kinematics
#               Given a JSON describing poses throughout two video views,
#               Extracts kinematics and computes kinematics through joint angles
#==================================================================================
#                                   Imports
#==================================================================================
import numpy as np
import json
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

#==================================================================================
#                                   Methods
#==================================================================================
# Calculates joint angle of knee
def calc_knee_angle(hip, knee, ankle, rightNeg):
    if (hip == [-1, -1] or knee == [-1, -1] or ankle == [-1,-1]):
        return None  # returns this value as error code for no keypoint detection

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

    if(rightNeg and bc[0] < m_ba[0]): angle = - angle
    if (not rightNeg and bc[0] > m_ba[0]): angle = - angle
    return angle.tolist()

# Calculates joint angle of hip
def calc_hip_angle(hip, knee, rightNeg, isFlex):
    if(hip == [-1,-1] or knee == [-1,-1]):
        return None # returns this value as error code for no keypoint detection

    # Identifying joint positions
    a = np.array(hip) # Main joint
    b = np.array(knee)

    # Compute vectors from joints
    ab = b - a
    m_N = np.array([0,-1])

    cosine_angle = np.dot(ab, m_N) / (np.linalg.norm(ab) * np.linalg.norm(m_N))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)

    if (rightNeg and ab[0] > m_N[0]): angle = - angle
    if (not rightNeg and ab[0] < m_N[0]): angle = - angle

    if(isFlex): angle = angle * 4/3 # A heuristic for catering for forward/backward pelvic tilt
    return angle.tolist()

# TODO: Remove: noticing redundancy, smoothing deals with outliers
# If angle to be fed in is an outlier, simply return the same angle value as before
def outlier_check(angle_list, new_angle):
    if(len(angle_list) == 0): return new_angle
    if(new_angle == None or angle_list[-1] == None): return new_angle

    angle_before = angle_list[-1]
    diff = 30
    if(new_angle > angle_before + diff or new_angle < angle_before - diff):
        return angle_before
    else:
        return new_angle

# Traversing through pose to compute kinematics
def raw_angles(data, rightNeg=False, limit=10000, invert = False, isFlex=False):

    knee_ang_L = []
    knee_ang_R = []
    hip_ang_L = []
    hip_ang_R = []

    count = 1
    for pose in data:
        #Left
        knee_L = pose[ptID['knee_L']]
        ankle_L = pose[ptID['ankle_L']]
        hip_L = pose[ptID['hip_L']]

        angle = calc_knee_angle(hip_L, knee_L, ankle_L, rightNeg)
        angle = outlier_check(knee_ang_L, angle)
        knee_ang_L.append(angle)
        angle = calc_hip_angle(hip_L, knee_L, rightNeg, isFlex)
        angle = outlier_check(hip_ang_L, angle)
        hip_ang_L.append(angle)

        #Right
        knee_R = pose[ptID['knee_R']]
        ankle_R = pose[ptID['ankle_R']]
        hip_R = pose[ptID['hip_R']]

        if(invert): angle = calc_knee_angle(hip_R, knee_R, ankle_R, not rightNeg)
        else: angle = calc_knee_angle(hip_R, knee_R, ankle_R, rightNeg)
        angle = outlier_check(knee_ang_R, angle)
        knee_ang_R.append(angle)

        if(invert): angle = calc_hip_angle(hip_R, knee_R, not rightNeg, isFlex)
        else: angle = calc_hip_angle(hip_R, knee_R, rightNeg, isFlex)
        angle = outlier_check(hip_ang_R, angle)
        hip_ang_R.append(angle)

        if(count == limit): break
        count += 1

    knee_ang = [knee_ang_L, knee_ang_R]
    hip_ang = [hip_ang_L, hip_ang_R]

    return knee_ang, hip_ang

# Checks which direction gait is from side view (affects how angles in saggital plane are calculated)
def checkGaitDirectionS(dataS, dimS):
    # Finds first instance of ankle in video
    for pose in dataS:
        ankle_L = pose[ptID['ankle_L']]
        ankle_R = pose[ptID['ankle_R']]
        if(ankle_L != [-1,-1]):
            ankle_init = ankle_L
            break
        if (ankle_R != [-1, -1]):
            ankle_init = ankle_R
            break

    init_x = ankle_init[0]
    max_x = dimS[0]
    if (init_x > max_x / 2):
        return True
    else:
        return False

# Computes and saves kinematics (joint angles) from poses
def kinematics_extract(readFile, writeFile):
    with open(readFile, 'r') as f:
        jsonPose = json.load(f)

    jsonList = []
    for cap in jsonPose:
        dataS = cap['dataS']
        dimS = cap['dimS']
        dataF = cap['dataF']
        lenS = cap['lenS']
        lenF = cap['lenF']

        limit = max(lenF, lenS) # Can set to min if the same is desired
        rightNeg = checkGaitDirectionS(dataS, dimS) # True: Right to Left, False: Left to Right

        knee_FlexExt, hip_FlexExt = raw_angles(dataS, rightNeg, limit, isFlex=True)
        knee_AbdAdd, hip_AbdAdd = raw_angles(dataF, rightNeg, limit=limit, invert=True)
        jsonDict = {
            'knee_FlexExt' : knee_FlexExt,
            'hip_FlexExt' : hip_FlexExt,
            'knee_AbdAdd' : knee_AbdAdd,
            'hip_AbdAdd' : hip_AbdAdd
        }
        jsonList.append(jsonDict)

    with open(writeFile, 'w') as outfile:
        json.dump(jsonList, outfile, separators=(',', ':'))

#==================================================================================
#                                   Main
#==================================================================================
path = '..\\Part10\\'
readFile = path + 'Part10_pose.json'
writeFile = path + 'Part10_angles.json'
start_time = time.time()
kinematics_extract(readFile, writeFile)
print('Kinematics extracted and saved in', '\"'+writeFile+'\"', '[Time:', '{0:.2f}'.format(time.time() - start_time), 's]')

path = '..\\Part11\\'
readFile = path + 'Part11_pose.json'
writeFile = path + 'Part11_angles.json'
start_time = time.time()
kinematics_extract(readFile, writeFile)
print('Kinematics extracted and saved in', '\"'+writeFile+'\"', '[Time:', '{0:.2f}'.format(time.time() - start_time), 's]')

path = '..\\Part12\\'
readFile = path + 'Part12_pose.json'
writeFile = path + 'Part12_angles.json'
start_time = time.time()
kinematics_extract(readFile, writeFile)
print('Kinematics extracted and saved in', '\"'+writeFile+'\"', '[Time:', '{0:.2f}'.format(time.time() - start_time), 's]')
