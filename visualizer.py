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

# TODO: Scrap, and make version that works strictly with json
def plot_debug(img, coords, confidence, class_ids, bboxes, scores,
                   box_thresh=0.5, keypoint_thresh=0.2):

    if isinstance(coords, mx.nd.NDArray):
        coords = coords.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if isinstance(confidence, mx.nd.NDArray):
        confidence = confidence.asnumpy()

    joint_visible = confidence[:, :, 0] > keypoint_thresh
    joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                   [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]]

    bbox_first = bboxes[0][0]
    x_min = bbox_first[0]
    y_min = bbox_first[1]
    x_max = bbox_first[2]
    y_max= bbox_first[3]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) # img.shape[1], img.shape[0] (BELOW)
    ax.set(xlim=(0, img.shape[1]), ylim = (0, img.shape[0])) # setting width and height of plot
    #ax.invert_yaxis()

    i = scores.argmax() # gets index of most confident bbox estimation
    colormap_index = np.linspace(0, 1, len(joint_pairs))
    pts = coords[i]

    for cm_ind, jp in zip(colormap_index, joint_pairs):
        if joint_visible[i, jp[0]] and joint_visible[i, jp[1]]:

            ax.plot(pts[jp, 0] - x_min, y_max - pts[jp, 1],
                    linewidth=3.0, alpha=0.7, color=plt.cm.cool(cm_ind))
            ax.scatter(pts[jp, 0] - x_min, y_max - pts[jp, 1], s=20)

    return ax