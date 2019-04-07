import os
import time
import sys

import tensorflow as tf
import numpy as np
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PIL import Image


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


# Define the video stream
# What model to download.
MODEL_DIR = './frozen_model'
MODEL = 'ssdlite_mobilenet_v2_coco'
DATA_DIR = './data/'
CONFIG_FILE = MODEL + '.config'
CHECKPOINT_FILE = 'model.ckpt'
PATH_TO_LABELS = 'training/data/object_detection.pbtxt'
OPTIMIZED_MODEL_FILE = 'optimized_model.pbtxt'

PIPELINE_CONFIG_NAME='pipeline.config'
CHECKPOINT_PREFIX='model.ckpt'

# Number of classes to detect
NUM_CLASSES = 5
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
tf_input = tf_sess.graph.get_tensor_by_name('prefix/image_tensor:0')
tf_scores = tf_sess.graph.get_tensor_by_name('prefix/detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('prefix/detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('prefix/detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('prefix/num_detections:0')

files = os.listdir(sys.argv[1])

for f in files:
    image_np = np.array(Image.open(f))
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
    im = Image.fromarray(image_np)
    im.save(os.path.join(sys.argv[2], f))
