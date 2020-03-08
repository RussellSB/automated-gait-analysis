#==================================================================================
#                                   COMPARISON
#----------------------------------------------------------------------------------
#                      Input: Gait cycles, Output: Similarity
#               Compares the kinematics of my automated system with
#                       the lab's Vicon Plug-In-Gait System
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
# Compares using the two-sample Kolmogorov-Smirnof 2 sample test
def compare_ks(samp1, samp2, title, isLeft):
    test = stats.ks_2samp(samp1, samp2)
    samp1_size = len(samp1)
    samp2_size = len(samp2)

    isDifferent1 = False
    crit_value = 1.63 * math.sqrt((samp1_size + samp2_size) / (samp1_size * samp2_size))
    if(test.statistic > float(crit_value)):
        isDifferent1 = True

    isDifferent2 = False
    if(test.pvalue < 0.01):
        isDifferent1 = True

    LorR = 'L' if isLeft else 'R'
    pval = '%.2f' % test.pvalue
    d = '%.2f' % test.statistic
    cv = '%.2f' % crit_value

    print(title+'\t'+LorR+'\t\t'+pval+'\t'+str(0.01)+'\t'+d+'\t'+cv+'\t'+str(isDifferent1)[0]+'\t'+str(isDifferent2)[0])

# Compares textually for either flexion/extension or abduction/adduction with respect to average using spm1d
def compare_textually(gc_PE, gc_PIG, code):
    print('===========================================================================')
    print('Title'+'\t\t\t'+'L/R'+'\t\t'+'p'+'\t\t'+'a'+'\t\t'+'D'+'\t\t'+'cv'+'\t\t'+'M1'+'\t'+'M2')
    print('===========================================================================')
    # TODO: finish off

    dict = {}
    dict['pe_knee_L'] = gc_PE['knee_' + code + '_avg']['gcL_avg']
    dict['pig_knee_L'] = gc_PIG['knee_' + code + '_avg']['gcL_avg']
    dict['pe_knee_R'] = gc_PE['knee_' + code + '_avg']['gcR_avg']
    dict['pig_knee_R'] = gc_PIG['knee_' + code + '_avg']['gcR_avg']

    dict['pe_hip_L'] = gc_PE['hip_' + code + '_avg']['gcL_avg']
    dict['pig_hip_L'] = gc_PIG['hip_' + code + '_avg']['gcL_avg']
    dict['pe_hip_R'] = gc_PE['hip_' + code + '_avg']['gcR_avg']
    dict['pig_hip_R'] = gc_PIG['hip_' + code + '_avg']['gcR_avg']

    i = 0
    isLeft = True
    for key in dict:
        if (i % 2):  # COMPARE
            y1 = dict[key]
            title = key.split('_')[1] + '(' + key.split('_')[2] + ')' + code
            compare_ks(y0, y1, title, isLeft)

            isLeft = False if isLeft else True
        else:
            y0 = dict[key]
        i += 1
    print('===========================================================================\n')

# Compares using Statistical Parametric Mapping 1D 2 sample t-test
def compare_spm1d(y0, y1):
    alpha = 0.01
    t = spm1d.stats.ttest2(y0, y1, equal_var=False)
    ti = t.inference(alpha, two_tailed=False)
    return ti

# Compares visually for either flexion/extension or abduction/adduction with respect to instances using spm1d
def compare_visually(gc_PE, gc_PIG, code):
    dict = {}
    dict['pe_knee_L'] = gc_PE['knee_'+code+'_gc'][0]
    dict['pig_knee_L'] = gc_PIG['knee_'+code+'_gc'][0]
    dict['pe_knee_R'] = gc_PE['knee_'+code+'_gc'][1]
    dict['pig_knee_R'] = gc_PIG['knee_'+code+'_gc'][1]

    dict['pe_hip_L'] = gc_PE['hip_'+code+'_gc'][0]
    dict['pig_hip_L'] = gc_PIG['hip_'+code+'_gc'][0]
    dict['pe_hip_R'] = gc_PE['hip_'+code+'_gc'][1]
    dict['pig_hip_R'] = gc_PIG['hip_'+code+'_gc'][1]

    i = 0
    isLeft = True
    color = red
    for key in dict:
        if (i % 2):  # COMPARE
            y1 = np.array(dict[key])
            ti = compare_spm1d(y0, y1)

            plt.figure(figsize=(8, 3.5))
            ax = plt.axes((0.1, 0.15, 0.35, 0.8))
            spm1d.plot.plot_mean_sd(y0)
            spm1d.plot.plot_mean_sd(y1, linecolor=color, facecolor=color)
            ax.axhline(y=0, color='k', linestyle=':')

            ax.set_xlabel('Time (%)')
            title = key.split('_')[1] + ' (' + key.split('_')[2] + ')' + ' ' + code
            ax.set_ylabel(r"" + title + " ${\Theta}$ (degrees)")

            ax = plt.axes((0.55, 0.15, 0.35, 0.8))
            ti.plot()
            ti.plot_threshold_label(fontsize=8)
            ti.plot_p_values(size=10, offsets=[(0, 0.3)])
            ax.set_xlabel('Time (%)')
            plt.show()
            plt.close()

            isLeft = False if (isLeft == True) else True
            color = red if (isLeft) else blue
        else:
            y0 = np.array(dict[key])
        i += 1

#==================================================================================
#                                   Main
#==================================================================================
filePE = '..\\Part01\\' + 'Part01_gc.json'
filePIG = '..\\Part08\\' + 'Part08_gc.json' # For now, note he has weird knee abd/add

with open(filePE, 'r') as f:
    gc_PE = json.load(f)
with open(filePIG, 'r') as f:
    gc_PIG = json.load(f)

# Using Kolmogorov-Smirnoff on the instances
compare_textually(gc_PE, gc_PIG, 'FlexExt')
compare_textually(gc_PE, gc_PIG, 'AbdAdd')

# Using SPM1D on the instances
compare_visually(gc_PE, gc_PIG, 'FlexExt')
compare_visually(gc_PE, gc_PIG, 'AbdAdd')