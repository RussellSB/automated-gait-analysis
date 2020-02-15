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
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import imageio
import json

#==================================================================================
#                                   Constants
#==================================================================================
joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                   [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]]

#==================================================================================
#                                   Methods
#==================================================================================

# Plots the pose of one frame, and returns axis
def plot_pose(data, dim):
    for frame in data:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlim=(0, dim[0]), ylim=(0, dim[1]))  # setting width and height of plot
        colormap_index = np.linspace(0, 1, len(joint_pairs))
        for cm_ind, jp in zip(colormap_index, joint_pairs):
            joint1 = frame[jp[0]]
            joint2 = frame[jp[1]]
            if(joint1 > [-1, -1] and joint2 > [-1,-1]):
                print(joint1, joint2)
                x = [joint1[0], joint2[0]]
                y = [joint1[1], joint2[1]]
                ax.plot(x, y, linewidth=3.0, alpha=0.7, color=plt.cm.cool(cm_ind))
                ax.scatter(x, y, s=20)
        plt.show()

# Makes a gif out of what is described in the jsonFile
def jsonPose_to_gif(jsonFile):
    with open('test10.json', 'r') as f:
        jsonPose = json.load(f)

    dataS = jsonPose[0]['dataS']
    dimS = jsonPose[0]['dimS']

    dataF = jsonPose[0]['dataF']
    dimF = jsonPose[0]['dimF']


#jsonPose_to_gif('test10.json')