#==================================================================================
#                               VISUALIZER
#----------------------------------------------------------------------------------
#                           Input: JSON, Output: Pose plot GIFS
#               Visualizes saved graph structure of poses, shows GIFS
#               describing the saved points in action. Great for testing
#               that videos were pose estimated correctly, before feature
#               extraction.
#----------------------------------------------------------------------------------
#==================================================================================
# TODO Cater for large json files with more than one capture
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
# Saves every pose frame of the video
def save_poses(data, dim, path, limit):
    i = 1
    for frame in data:
        fig, ax = plt.subplots()
        ax.set(xlim=(0, dim[0]), ylim=(0, dim[1]))  # setting width and height of plot

        for cm_ind, jp in zip(colormap_index, joint_pairs):
            joint1 = frame[jp[0]]
            joint2 = frame[jp[1]]
            if (joint1 > [-1, -1] and joint2 > [-1, -1]):
                x = [joint1[0], joint2[0]]
                y = [joint1[1], joint2[1]]
                ax.plot(x, y, linewidth=3.0, alpha=0.7, color=plt.cm.cool(cm_ind))
                ax.scatter(x, y, s=20)
        filename = path + str(i) + '.png'
        plt.savefig(filename)
        if(i == limit): break
        i += 1

# Makes a collection of figures out of what is described in the jsonFile
def jsonPose_to_pics(jsonFile, path):
    with open(jsonFile, 'r') as f:
        jsonPose = json.load(f)

    lenF = jsonPose[0]['lenF']
    lenS = jsonPose[0]['lenS']
    limit = min(lenF, lenS)

    dataS = jsonPose[0]['dataS']
    dimS = jsonPose[0]['dimS']
    path1 = path + jsonPose[0]['id'] + '-S/'
    save_poses(dataS, dimS, path1, limit)

    dataF = jsonPose[0]['dataF']
    dimF = jsonPose[0]['dimF']
    path2 = path + jsonPose[0]['id'] + '-F/'
    save_poses(dataF, dimF, path2, limit)

path = '../Test/GIF/'
jsonPose_to_pics('test.json', path)