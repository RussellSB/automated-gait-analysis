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
from gluoncv.model_zoo import get_model
from tqdm import tqdm
import gluoncv as gcv
import mxnet as mx
import time, cv2
import json
import glob
import os

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
    if (cap.isOpened() == False): # Check if camera opened successfully
        print("Error opening video stream or file")
        return

    frame_count = 0
    pose_data_vid = []
    dimensions = (0, 0)
    frame_length = (int)(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=frame_length, ncols=1, desc='.')
    pbar.ncols = 100

    # Iterate through every frame in video
    while(cap.isOpened()):
        ret, frame = cap.read() # read current frame
        if (frame is None):
            break # If current frame doesn't exist, finished iterating through frames
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8') # mxnet readable

        # Object detection
        x, frame = gcv.data.transforms.presets.yolo.transform_test(frame)  # short=512, max_size=350
        class_IDs, scores, bounding_boxs = detector(x.as_in_context(ctx))

        # Pose estimation
        pose_input, upscale_bbox = detector_to_alpha_pose(frame, class_IDs, scores, bounding_boxs,
                                                          output_shape=(320, 256))
        # Gets current pose keypoints
        if (upscale_bbox is None): # Caters for no person detection
            pbar.set_description_str('Skipping  ')
            pose_data_curr = [[-1, -1] for j in range(0, 17)]
        else: # Caters for person detection
            pbar.set_description_str('Processing')
            predicted_heatmap = estimator(pose_input)
            pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

            scores = scores.asnumpy()
            confidence = confidence.asnumpy()
            pred_coords = pred_coords.asnumpy()

            # Preparing for json
            pose_data_curr = curr_pose(frame, pred_coords, confidence, scores, keypoint_thresh=0.2)
        pose_data_vid.append(pose_data_curr)

        if (frame_count == 0):
            dimensions = [frame.shape[1], frame.shape[0]]
        frame_count += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    return dimensions, pose_data_vid

# Given two videos it will output the json describing all poses in both videos
def videos_to_jsonPose(vidSide, vidFront, partId, capId, isNormal):
    dimensions_side, pose_vid_side = video_to_listPose(vidSide)
    dimensions_front, pose_vid_front = video_to_listPose(vidFront)

    if(dimensions_side != dimensions_front):
        print('Warning: side video', dimensions_side, 'and front video', dimensions_front, 'of different dimensions' )

    if (len(pose_vid_side) != len(pose_vid_front)):
        print('Warning: side video', len(pose_vid_side), 'and front video', len(pose_vid_front), 'of different frame counts')

    jsonPose_dict = {
        'partId': partId,
        'capId': capId,
        'normal': isNormal,
        'dimS': dimensions_side,
        'lenS' : len(pose_vid_side),
        'dimF' : dimensions_front,
        'lenF' : len(pose_vid_front),
        'dataS' : pose_vid_side,
        'dataF' : pose_vid_front
    }
    return jsonPose_dict

def estimate_poses(path, writeFile):
    jsonPose_list = []
    fs_pair = []
    i = 0
    for filename in glob.glob(os.path.join(path, '*.avi')):
        fs_pair.append(filename)
        if (i % 2):
            print('Capture pair', '('+ str(int((i+1)/2)) +'/6)', ':', '\"'+fs_pair[1]+'\"', ',', '\"'+fs_pair[0]+'\"')
            capId = fs_pair[0].split('-')[1]
            isNormalTag = fs_pair[0].split('-')[2]
            isNormal = True if isNormalTag == 'N' else False
            partId = fs_pair[0].split('-')[0].split('\\')[2]
            jsonPose_dict = videos_to_jsonPose(fs_pair[1], fs_pair[0], partId, capId, isNormal)
            jsonPose_list.append(jsonPose_dict)
            fs_pair.clear()
        i += 1

    with open(writeFile, 'w') as outfile:
        json.dump(jsonPose_list, outfile, separators=(',', ':'))

#==================================================================================
#                                   Main
#==================================================================================
path = '..\\Part03\\'
writeFile = path + 'Part03_pose.json'
start_time = time.time()
estimate_poses(path, writeFile)
print('Poses estimated and saved in', '\"'+writeFile+'\"', '[Time:', '{0:.2f}'.format(time.time() - start_time), 's]')

