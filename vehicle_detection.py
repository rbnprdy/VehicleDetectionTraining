import os
import time
import sys
import tensorflow as tf
import numpy as np
import cv2
#import tensorflow.contrib.tensorrt as trt
#from tf_trt_models.detection import download_detection_model
#from tf_trt_models.detection import build_detection_graph
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def open_cam_onboard(width, height):
    # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
    gst_str = ('nvcamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int)640, height=(int)480, '
               'format=(string)I420, framerate=(fraction)240/1 ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

WIDTH = 640
HEIGHT = 480

# Define the video stream
# cap = open_cam_onboard(WIDTH, HEIGHT)
cap = cv2.VideoCapture(sys.argv[1])

out = cv2.VideoWriter(sys.argv[2],cv2.VideoWriter_fourcc(*'MP4V'), 10, (WIDTH,HEIGHT))

# What model to download.
MODEL_DIR = './frozen_model'
MODEL = 'ssdlite_mobilenet_v2_coco'
DATA_DIR = './data/'
CONFIG_FILE = MODEL + '.config'
CHECKPOINT_FILE = 'model.ckpt'
#PATH_TO_LABELS = '../third_party/models/research/object_detection/data/' +\
#    'mscoco_label_map.pbtxt'
PATH_TO_LABELS = 'training/data/object_detection.pbtxt'
OPTIMIZED_MODEL_FILE = 'optimized_model.pbtxt'

PIPELINE_CONFIG_NAME='pipeline.config'
CHECKPOINT_PREFIX='model.ckpt'

# Number of classes to detect
NUM_CLASSES = 1
"""if not os.path.exists(os.path.join(DATA_DIR, OPTIMIZED_MODEL_FILE)):
    print('Creating optimized graph...')
    # Download model and build frozen graph
    # config_path, checkpoint_path = download_detection_model(MODEL, 'data')
    config_path = os.path.join(MODEL_DIR, PIPELINE_CONFIG_NAME)
    checkpoint_path = os.path.join(MODEL_DIR, CHECKPOINT_PREFIX)
    frozen_graph, input_names, output_names = build_detection_graph(
        config=config_path,
        checkpoint=checkpoint_path,
        score_threshold=0.3,
        batch_size=1
    )
    # frozen_graph = load_graph('frozen_model/frozen_inference_graph.pb')
    # Optimize with TensorRT
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP16',
        minimum_segment_size=50
    )
    with tf.gfile.GFile(
            os.path.join(DATA_DIR, OPTIMIZED_MODEL_FILE), 'wb') as f:
        f.write(trt_graph.SerializeToString())
else:
    print('Loaded optimized graph')
    trt_graph = tf.GraphDef()
    with open(os.path.join(DATA_DIR, OPTIMIZED_MODEL_FILE), 'rb') as f:
        trt_graph.ParseFromString(f.read())
"""
graph = load_graph('frozen_model/frozen_inference_graph.pb')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

tf_sess = tf.Session(config=tf_config, graph=graph)
#for op in (tf_sess.graph.get_operations()):
#    print(op.name)
#quit()
# tf.import_graph_def(trt_graph, name='')
#tf.import_graph_def(graph, name='')
tf_input = tf_sess.graph.get_tensor_by_name('prefix/image_tensor:0')
tf_scores = tf_sess.graph.get_tensor_by_name('prefix/detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('prefix/detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('prefix/detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('prefix/num_detections:0')

# Number of iterations
num_iterations = 1000

total_time = 0

for i in range(0, num_iterations):
    print('{}/{}'.format(i, num_iterations))
    start_time = time.time()
    # Read frame from camera
    ret, image_np = cap.read()
    if not ret:
        break
    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes, scores, classes, num_detections) = tf_sess.run(
        [tf_boxes, tf_scores, tf_classes, tf_num_detections],
        feed_dict={tf_input: image_np_expanded})
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    # Calculate FPS
    elapsed_time = time.time() - start_time
    #cv2.putText(image_np, '%.3g fps' % (1 / elapsed_time),
    #            (10, 30),
    #            cv2.FONT_HERSHEY_SIMPLEX,
    #            1,
    #            (255, 255, 255),
    #            2)
    # Display output
    # cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
    # cv2.imshow('object detection', image_np)
    out.write(image_np)
    total_time += elapsed_time

cap.release()
out.release()

print('average fps: {}'.format(num_iterations / total_time))
