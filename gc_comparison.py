#==================================================================================
#                                   COMPARISON
#----------------------------------------------------------------------------------
#                      Input: Gait cycles, Output: Similarity
#               Compares the kinematics of my automated system with
#                       the lab's Vicon Plug-In-Gait System
#==================================================================================
#                                   Imports
#==================================================================================
import json
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math
from statistics import mean

#==================================================================================
#                                   Constants
#==================================================================================
red = "#ff002b" # "#FF4A7E"# "#E0082D"
blue =  "#0077ff" # "#72B6E9" # "#55BED7"#
red2 =  "#383838" #"#682c8e" #"#ff4800"# "#ff002b"
blue2 = "#383838" # "#682c8e" # "#0077ff"

#==================================================================================
#                                   Methods
#==================================================================================

# Compare all visually through graph plots
def compare_visually_all(gc_PE, gc_PIG, code, name):
    dictAvg = {}
    dictAvg['pe_knee_L'] = gc_PE['knee_' + code + '_avg']['gcL_avg']
    dictAvg['pig_knee_L'] = gc_PIG['knee_' + code + '_avg']['gcL_avg']
    dictAvg['pe_knee_R'] = gc_PE['knee_' + code + '_avg']['gcR_avg']
    dictAvg['pig_knee_R'] = gc_PIG['knee_' + code + '_avg']['gcR_avg']

    dictAvg['pe_hip_L'] = gc_PE['hip_' + code + '_avg']['gcL_avg']
    dictAvg['pig_hip_L'] = gc_PIG['hip_' + code + '_avg']['gcL_avg']
    dictAvg['pe_hip_R'] = gc_PE['hip_' + code + '_avg']['gcR_avg']
    dictAvg['pig_hip_R'] = gc_PIG['hip_' + code + '_avg']['gcR_avg']

    dictSTD = {}
    dictSTD['pe_knee_L'] = gc_PE['knee_' + code + '_avg']['gcL_std']
    dictSTD['pig_knee_L'] = gc_PIG['knee_' + code + '_avg']['gcL_std']
    dictSTD['pe_knee_R'] = gc_PE['knee_' + code + '_avg']['gcR_std']
    dictSTD['pig_knee_R'] = gc_PIG['knee_' + code + '_avg']['gcR_std']

    dictSTD['pe_hip_L'] = gc_PE['hip_' + code + '_avg']['gcL_std']
    dictSTD['pig_hip_L'] = gc_PIG['hip_' + code + '_avg']['gcL_std']
    dictSTD['pe_hip_R'] = gc_PE['hip_' + code + '_avg']['gcR_std']
    dictSTD['pig_hip_R'] = gc_PIG['hip_' + code + '_avg']['gcR_std']

    print("PE: {:.0f}L, {:.0f}R".format(
        gc_PE['knee_' + code + '_avg']['gcL_count'], gc_PE['knee_' + code + '_avg']['gcR_count']))
    print("PIG: {:.0f}L, {:.0f}R".format(
        gc_PIG['knee_' + code + '_avg']['gcL_count'], gc_PIG['knee_' + code + '_avg']['gcR_count']))

    isLeft = True
    color=red
    color2=red2
    i = 0
    for key in dictAvg:
        if (i % 2):  #  compares here
            y1 = dictAvg[key]
            y1STD = dictSTD[key]

            fig, ax = plt.subplots()
            side = 'Left ' if(isLeft) else 'Right '
            title = side + (key.split('_')[1]).capitalize() + ' ' + name
            ax.set_title(title)
            ax.set_xlabel('Time (%)')  #
            ax.set_ylabel(r"${\Theta}$ (degrees)")

            ax.plot(y0, color=color, label='Automated') # mean PE
            std1 = (np.array(y0) + np.array(y0STD)).tolist()
            std2 = (np.array(y0) - np.array(y0STD)).tolist()
            ax.plot(std1, '--', color=color, alpha=0)
            ax.plot(std2, '--', color=color, alpha=0)
            ax.fill_between(range(0,101), std1, std2, color=color, alpha=0.15)

            ax.plot(y1, color=color2, label='Marker-based') # mean PIG
            std1 = (np.array(y1) + np.array(y1STD)).tolist()
            std2 = (np.array(y1) - np.array(y1STD)).tolist()
            ax.plot(std1, '--', color=color2, alpha=0)
            ax.plot(std2, '--', color=color2, alpha=0)
            ax.fill_between(range(0,101), std1, std2, color=color2, alpha=0.15)

            plt.xlim(0, 100)
            ax.legend()
            plt.show()
            plt.close()

            isLeft = False if isLeft else True
            color = red if (isLeft) else blue
            color2 = red2 if (isLeft) else blue2
        else:
            y0 = dictAvg[key]
            y0STD = dictSTD[key]
        i += 1

def errors_all(gc_PE, gc_PIG, code, name):
    dictAvg = {}
    dictAvg['pe_knee_L'] = gc_PE['knee_' + code + '_avg']['gcL_avg']
    dictAvg['pig_knee_L'] = gc_PIG['knee_' + code + '_avg']['gcL_avg']
    dictAvg['pe_knee_R'] = gc_PE['knee_' + code + '_avg']['gcR_avg']
    dictAvg['pig_knee_R'] = gc_PIG['knee_' + code + '_avg']['gcR_avg']

    dictAvg['pe_hip_L'] = gc_PE['hip_' + code + '_avg']['gcL_avg']
    dictAvg['pig_hip_L'] = gc_PIG['hip_' + code + '_avg']['gcL_avg']
    dictAvg['pe_hip_R'] = gc_PE['hip_' + code + '_avg']['gcR_avg']
    dictAvg['pig_hip_R'] = gc_PIG['hip_' + code + '_avg']['gcR_avg']

    i = 0
    overall_error = []
    for key in dictAvg:
        if (i % 2):  # COMPARE
            y1 = dictAvg[key] # MB

            error = []
            for n in range(len(y1)):
                error.append(abs(y1[n] - y0[n]))
            overall_error += error
            print("{}\t\t{:.2f}\t{:.2f}\t{:.2f}".format(code+'  '+key[4:], min(error), mean(error), max(error)))

        else:
            y0 = dictAvg[key] # PE
        i += 1

    return overall_error

#==================================================================================
#                                   Main
#==================================================================================
def main():
    i = '14'  # Set participant you want to compare here (Either 1, 3, 5 or 14)
    filePath = '..\\Part'+ i + '\\'
    filePE = filePath + 'Part' + i + '_gc.json'
    filePIG = filePath + 'Part' + i + '_gc_pig.json'
    with open(filePE, 'r') as f:
        gc_PE = json.load(f)
    with open(filePIG, 'r') as f:
        gc_PIG = json.load(f)

    # Simply displaying them visually by mean and standard deviation
    compare_visually_all(gc_PE, gc_PIG, 'FlexExt', 'Flexion and Extension')
    compare_visually_all(gc_PE, gc_PIG, 'AbdAdd', 'Abduction and Adduction')

    print('KinematicVariable \tMin \tAvg \tMax')
    overall_error = []
    overall_error += errors_all(gc_PE, gc_PIG, 'FlexExt', 'Flexion and Extension')
    overall_error += errors_all(gc_PE, gc_PIG, 'AbdAdd', 'Abduction and Adduction')
    print('=======================================')
    print("Average angle error overall: {:.2f}".format(mean(overall_error)))

if __name__ == '__main__':
    main()