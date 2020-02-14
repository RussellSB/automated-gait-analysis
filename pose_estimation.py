#==================================================================================
#                               POSE ESTIMATION
#----------------------------------------------------------------------------------
#                           Input: Video x 2, Output: JSON
#               Given a front/back view and side view video of someone
#               walking, this will generate a json, describing the pose
#               via key-points in graph form, throughout every frame.
#----------------------------------------------------------------------------------
#==================================================================================
# TODO: Later expand to store more than just one pair of videos per person (for average)

#==================================================================================
#                                   Imports
#==================================================================================
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints
from gluoncv.model_zoo import get_model
from __future__ import division
import matplotlib.pyplot as plt
import gluoncv as gcv
import numpy as np
import mxnet as mx
import time, cv2
import json

#==================================================================================
#                                   Methods
#==================================================================================


# Given two videos it will output the json describing all poses in both videos
def videos_to_jsonPose(vidSide, vidFront):
    print('init')


#==================================================================================
#                                   Main
#==================================================================================
path = '../Test/'
vidSide = path + 'Part01test-side'
vidFront = path + 'Part01test-front'
videos_to_jsonPose(vidSide, vidFront)