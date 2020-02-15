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
from __future__ import division
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints
from gluoncv.model_zoo import get_model
import matplotlib.pyplot as plt
import gluoncv as gcv
import numpy as np
import mxnet as mx
import time, cv2
import json

#==================================================================================
#                                   AI Detectors
#                        Object detector; YOLO via GPU.
#                Pose Estimator; AlphaPose via CPU (no GPU support).
#==================================================================================
ctx = mx.gpu(0)
detector = get_model('yolo3_mobilenet1.0_coco', pretrained=True, ctx=ctx)
detector.reset_class(classes=['person'], reuse_weights={'person': 'person'})
estimator = get_model('alpha_pose_resnet101_v1b_coco', pretrained='de56b871')
detector.hybridize()
estimator.hybridize()

#==================================================================================
#                                   Methods
#==================================================================================
# Returns a keypoints with respect to the current frame's pose
def curr_pose(img, coords, confidence, scores, keypoint_thresh=0.2):
    i = scores.argmax() # gets index of most confident bbox estimation
    pts = coords[i] # coords of most confident pose in frame

    pose_data = []

    for j in range(0, len(pts)):
        x = -1
        y = -1
        if(confidence[i][j] > keypoint_thresh):
            x = int(pts[j][0])
            y = int(img.shape[0] - pts[j][1])
        pose_data.append([x,y])

    return pose_data

# Given one video, returns list of pose information in preparation for json file
def video_to_listPose(vid):
    cap = cv2.VideoCapture(vid) # load video
    frame_count = 0
    pose_data_vid = []
    dimensions = (0, 0)
    frame_length = (int)(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Iterate through every frame in video
    while(cap.isOpened()):
        ret, frame = cap.read() # read current frame
        if (frame is None):
            break # If current frame doesn't exist, finished iterating through frames
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8') # mxnet readable

        # Object detection
        x, frame = gcv.data.transforms.presets.yolo.transform_test(frame)  # short=512, max_size=350
        class_IDs, scores, bounding_boxs = detector(x.as_in_context(ctx))
        # Stores xyxy of the bounding box of the first frame

        # Pose estimation
        pose_input, upscale_bbox = detector_to_alpha_pose(frame, class_IDs, scores, bounding_boxs,
                                                          output_shape=(320, 256))
        if (upscale_bbox is None):
            break  # If no person detected in current frame, halt the analysis

        predicted_heatmap = estimator(pose_input)
        pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

        scores = scores.asnumpy()
        confidence = confidence.asnumpy()
        pred_coords = pred_coords.asnumpy()

        # Preparing for json
        pose_data_curr = curr_pose(frame, pred_coords, confidence, scores, keypoint_thresh=0.2)
        pose_data_vid.append(pose_data_curr)
        if(frame_count == 0):
            dimensions = [frame.shape[1], frame.shape[0]]

        print('Processing:', vid, ':', frame_count + 1, '/', frame_length)
        frame_count += 1
    cap.release()

    return dimensions, pose_data_vid

# Given two videos it will output the json describing all poses in both videos
def videos_to_jsonPose(vidSide, vidFront, isNormal):
    dimensions_side, pose_vid_side = video_to_listPose(vidSide)
    dimensions_front, pose_vid_front = video_to_listPose(vidFront)

    if(dimensions_side != dimensions_front):
        print('Warning: side video', dimensions_side, 'and front video', dimensions_front, 'of different dimensions' )

    if (len(pose_vid_side) != len(pose_vid_front)):
        print('Warning: side video', len(pose_vid_side), 'and front video', len(pose_vid_front), 'of different frame counts')

    jsonPose_list = []
    jsonPose_dict = {
        'id': 'test',
        'normal': isNormal,
        'dimS': dimensions_side,
        'lenS' : len(pose_vid_side),
        'dimF' : dimensions_front,
        'lenF' : len(pose_vid_front),
        'dataS' : pose_vid_side,
        'dataF' : pose_vid_front
    }
    jsonPose_list.append(jsonPose_dict)

    # TODO: Extract filename from vid name
    with open('test10' + '.json', 'w') as outfile:
        json.dump(jsonPose_list, outfile, separators=(',', ':'))

#==================================================================================
#                                   Main
#==================================================================================
path = '../Test/'
vidCoffee = path + 'coffee.mp4'
vidSide = path + 'Part01test-side.avi'
vidFront = path + 'Part01test-front.avi'

start_time = time.time()
videos_to_jsonPose(vidSide, vidFront,True)
print('JSON pose file generated:', '{0:.2f}'.format(time.time() - start_time), 's')