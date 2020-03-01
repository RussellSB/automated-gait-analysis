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
    ax.set_xlabel('Time (%)')
    ax.set_ylabel(r"${\Theta}$ (degrees)")
    ax.set(xlim=(0, xmax), ylim=(yrange[0], yrange[1]))

    for i in range(0, len(angleLists)):
        ax.plot(angleLists[i])

    plt.show()
    plt.close()

# Compares using the two-sample Kolmogorov-Smirnof 2 sample test
def comparison1(samp1, samp2):
    test = stats.ks_2samp(samp1, samp2)
    samp1_size = len(samp1)
    samp2_size = len(samp2)

    crit_value = 1.63 * math.sqrt((samp1_size + samp2_size) / (samp1_size * samp2_size))
    if(test.statistic > float(crit_value)):
        print("D =", test.statistic, ">", crit_value, "\n\tTherefore kinematics differ significantly")
    else:
        print("D =", test.statistic, "<", crit_value, "\n\tTherefore kinematics do not differ significantly")

    if(test.pvalue < 0.01):
        print("P-value =", test.pvalue, "< 0.01", "\n\tTherefore kinematics differ significantly")
    else:
        print("P-value =", test.pvalue, "> 0.01", "\n\tTherefore kinematics do not differ significantly")

# Compares using Statistical Parametric Mapping 1D 2 sample t-test
def comparison2(y0, y1, title):
    alpha = 0.01
    t = spm1d.stats.ttest2(y0, y1, equal_var=False)
    ti = t.inference(alpha, two_tailed=False)

    plt.figure(figsize=(8, 3.5))
    ax = plt.axes((0.1, 0.15, 0.35, 0.8))
    spm1d.plot.plot_mean_sd(y0)
    spm1d.plot.plot_mean_sd(y1, linecolor='r', facecolor='r')
    ax.axhline(y=0, color='k', linestyle=':')

    ax.set_xlabel('Time (%)')
    ax.set_ylabel(r""+title+" ${\Theta}$ (degrees)")

    ax = plt.axes((0.55, 0.15, 0.35, 0.8))
    ti.plot()
    ti.plot_threshold_label(fontsize=8)
    ti.plot_p_values(size=10, offsets=[(0, 0.3)])
    ax.set_xlabel('Time (%)')
    plt.show()
    plt.close()
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

# Using Kolmogorof-Smirnof test on the average
y_range = (-20, 80)
angles1 = gc_PE['knee_FlexExt_avg']['gcL_avg']
angles2 = gc_PIG['knee_FlexExt_avg']['gcL_avg']
noise = np.random.randint(y_range[0], y_range[1], 101)
flat = np.random.randint(0, 1, 101)
samples = [angles1, angles2, noise, flat]
plot_comparison(samples, 'Knee Flexion/Extension', y_range) # Plotting
comparison1(samples[0], samples[1]) # Comparing

# Using SPM1D on the instances
angles2 = gc_PIG['knee_FlexExt_gc'][0]
y0 = np.array(angles2)
angles1 = gc_PE['knee_FlexExt_gc'][0]
y1 = np.array(angles1)
comparison2(y0, y1, 'Knee Flexion/Extension')