import time
import PySimpleGUI as sg
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from load_custom_model import detect_fn

CUSTOM_MODEL_NAME = 'my_efficentdet_d2_LuggageTagsAndGloves11-26'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    "WORKSPACE_PATH": os.path.join("Tensorflow", "workspace"),
    "SCRIPTS_PATH": os.path.join("Tensorflow", "scripts"),
    "APIMODEL_PATH": os.path.join("Tensorflow", "models"),
    "ANNOTATION_PATH": os.path.join("Tensorflow", "workspace", "annotations"),
    "IMAGE_PATH": os.path.join("Tensorflow", "workspace", "images"),
    "MODEL_PATH": os.path.join("Tensorflow", "workspace", "models"),
    "PRETRAINED_MODEL_PATH": os.path.join(
        "Tensorflow", "workspace", "pre-trained-models"
    ),
    "CHECKPOINT_PATH": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME
    ),
    "OUTPUT_PATH": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "export"
    ),
    "TFJS_PATH": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "tfjsexport"
    ),
    "TFLITE_PATH": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "tfliteexport"
    ),
    "PROTOC_PATH": os.path.join("Tensorflow", "protoc"),
}

files = {
    "PIPELINE_CONFIG": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "pipeline.config"
    ),
    "TF_RECORD_SCRIPT": os.path.join(paths["SCRIPTS_PATH"], TF_RECORD_SCRIPT_NAME),
    "LABELMAP": os.path.join(paths["ANNOTATION_PATH"], LABEL_MAP_NAME),
}

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
# Camera Settings
camera_Width  = 480 # 320 # 480 # 720 # 1080 # 1620
camera_Heigth = 360 # 240 # 360 # 540 # 810  # 1215
frameSize = (camera_Width, camera_Heigth)
video_capture1 = cv.VideoCapture(0)
# video_capture2 = cv.VideoCapture(1)
# video_capture3 = cv.VideoCapture(2)
# video_capture4 = cv.VideoCapture(3)
# video_capture5 = cv.VideoCapture(4)
time.sleep(2.0)

# init Windows Manager
sg.theme("DarkBlue")

# def webcam col
colwebcam1_layout = [[sg.Text("Camera 1 (Front Driver)", size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="cam1")]]
colwebcam1 = sg.Column(colwebcam1_layout, element_justification='center')

colwebcam2_layout = [[sg.Text("Camera 2 (Front Passenger)", size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="cam2")]]
colwebcam2 = sg.Column(colwebcam2_layout, element_justification='center')

colwebcam3_layout = [[sg.Text("Camera 3 (Driver Side)", size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="cam3")]]
colwebcam3 = sg.Column(colwebcam3_layout, element_justification='center')

colwebcam4_layout = [[sg.Text("Camera 4 (Passenger Side)", size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="cam4")]]
colwebcam4 = sg.Column(colwebcam4_layout, element_justification='center')

colwebcam5_layout = [[sg.Text("Camera 5 (Rear)", size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="cam5")]]
colwebcam5 = sg.Column(colwebcam5_layout, element_justification='center')

colslayout = [[colwebcam1, colwebcam2], [colwebcam3, colwebcam4], [colwebcam5]]

layout = [colslayout]

window    = sg.Window("FOD Detection", layout,
                    no_titlebar=False, alpha_channel=1, grab_anywhere=False,
                    return_keyboard_events=True, location=(100, 100))
while True:
    start_time = time.time()
    event, values = window.read(timeout=20)

    if event == sg.WIN_CLOSED:
        break

    ret, frame = video_capture1.read()
    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
    # Get rid of '##' to add the other cameras
    # Will crash if there are no other cameras in use

    # get camera frame for each camera
    #1
    ret, frameOrig1 = video_capture1.read()
    frame1 = cv.resize(image_np_with_detections, frameSize)

    ret, frame = video_capture1.read()
    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
    #2
    # ret, frameOrig2 = video_capture2.read()
    # frame2 = cv.resize(image_np_with_detections, frameSize)

    #3
    ##ret, frameOrig3 = video_capture3.read()
    ##frame3 = cv2.resize(frameOrig3, frameSize)
    
    #4
    ##ret, frameOrig4 = video_capture4.read()
    ##frame4 = cv2.resize(frameOrig4, frameSize)
    
    #5
    ##frame5 = cv2.resize(frameOrig5, frameSize)

    # # update webcam1
    imgbytes = cv.imencode(".png", frame1)[1].tobytes()
    window["cam1"].update(data=imgbytes)

    # # update webcam2
    imgbytes = cv.imencode(".png", frame1)[1].tobytes()
    window["cam2"].update(data=imgbytes)

    # # update webcam3
    ##imgbytes = cv2.imencode(".png", frame3)[1].tobytes()
    ##window["cam3"].update(data=imgbytes)

    # # update webcam4
    ##imgbytes = cv2.imencode(".png", frame4)[1].tobytes()
    ##window["cam4"].update(data=imgbytes)

    # # update webcam5
    ##imgbytes = cv2.imencode(".png", frame5)[1].tobytes()
    ##window["cam5"].update(data=imgbytes)

video_capture1.release()
video_capture2.release()
video_capture3.release()
video_capture4.release()
video_capture5.release()
cv.destroyAllWindows()
