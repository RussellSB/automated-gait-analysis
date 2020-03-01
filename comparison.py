#==================================================================================
#                                   COMPARISON
#----------------------------------------------------------------------------------
#                      Input: Gait cycles, Output: Similarity
#               Compares the kinematics of my automated system with
#                       the lab's Vicon Plug-In-Gait System
#----------------------------------------------------------------------------------
#==================================================================================
#==================================================================================
#                                   Imports
#==================================================================================
import spm1d
import json
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math

#==================================================================================
#                                   Constants
#==================================================================================
red = "#FF4A7E"
blue = "#72B6E9"

#==================================================================================
#                                   Methods
#==================================================================================
# Plots all angles in list of angle lists to compare
def plot_comparison(angleLists, title, yrange):
    xmax = 100
    fig, ax = plt.subplots()
    ax.set_title('Comparsion of ' + title)
    ax.set_xlabel('Data points')
    ax.set_ylabel(r"${\Theta}$ (degrees)")
    ax.set(xlim=(0, xmax), ylim=(yrange[0], yrange[1]))

    for i in range(0, len(angleLists)):
        ax.plot(angleLists[i])

    plt.show()
    plt.close()

# Compares using the two-sample Kolmogorov-Smirnof test
def ks(samp1, samp2):
    test = stats.ks_2samp(samp1, samp2)
    samp1_size = len(samp1)
    samp2_size = len(samp2)

    crit_value = 1.63 * math.sqrt((samp1_size + samp2_size) / (samp1_size * samp2_size))
    print(test.statistic, crit_value)
    if(test.statistic > float(crit_value)):
        print("D =", test.statistic, ">", crit_value, "\n\tTherefore kinematics differ significantly")
    else:
        print("D =", test.statistic, "<", crit_value, "\n\tTherefore kinematics do not differ significantly")

    if(test.pvalue < 0.01):
        print("P-value =", test.pvalue, "< 0.01", "\n\tTherefore kinematics differ significantly")
    else:
        print("P-value =", test.pvalue, "> 0.01", "\n\tTherefore kinematics do not differ significantly")

#==================================================================================
#                                   Main
#==================================================================================
path = '..\\Part01\\'
filePE = path + 'Part01_gc.json'
filePIG = '..\\Part02\\' + 'Part02_gc.json' # For now

with open(filePE, 'r') as f:
    gc_PE = json.load(f)
with open(filePIG, 'r') as f:
    gc_PIG = json.load(f)

# Initializing
y_range = (-20, 80)
angle1 = gc_PE['knee_FlexExt_avg']['gcL_avg']
angle2 = gc_PIG['knee_FlexExt_avg']['gcL_avg']
noise = np.random.randint(y_range[0], y_range[1], 101)
flat = np.random.randint(0, 1, 101)
angleLists = [angle1, angle2, noise, flat]

plot_comparison(angleLists, 'Knee Flexion/Extension', y_range) # Plotting
ks(angleLists[0], angleLists[1]) # Comparing


