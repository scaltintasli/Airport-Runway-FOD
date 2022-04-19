import time
import operator
import PySimpleGUI as sg
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import playsound
import tensorflow as tf
import folium
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import base64
import webbrowser
import folium
from folium import IFrame
from Detection import Detection
import os, shutil
from extract_coordinates import *
from threading import Thread
from tracker import *


def create_folder(folderName):
    exists = os.path.exists(folderName)
    if not exists:
        os.makedirs(folderName)

def delete_folder_contents(folderPath):
    for filename in os.listdir(folderPath):
        file_path = os.path.join(folderPath, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    print("Contents of folder deleted: " + folderPath)

def clearLog():
    with open("Log.txt", "a") as text_file:
        text_file.truncate(0)


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def findScore(scoreValues, threshold):
    found = [i for i, e in enumerate(scoreValues) if e >= threshold]
    return found

def getThreshold(threshString):
    return int(threshString[:-1]) / 100

def getCameraAmount(cameraString):
    return int(cameraString)

def createMap():
    m = folium.Map(location=starting_coords, zoom_start=20)

    tile = folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    return m

def openMap():
    detections_list = tracker.detections_list
    for det in detections_list:
        det.addPoint()

    webbrowser.open("map.html")

def tfBoundingBoxes(frame, detectionKey, detectionKey2, threshold):
    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Used for object tracking
    listDetections = []

    #List of objects position in detections[] with a certain threshold
    positionList = findScore(detections['detection_scores'], threshold)

    ### Old way of creating labels ###
    # label_id_offset = 1
    # image_np_with_detections = image_np.copy()
    #
    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #             image_np_with_detections,
    #             detections['detection_boxes'],
    #             detections['detection_classes']+label_id_offset,
    #             detections['detection_scores'],
    #             category_index,
    #             use_normalized_coordinates=True,
    #             max_boxes_to_draw=5,
    #             min_score_thresh=threshold,
    #             agnostic_mode=False)

    # Gets the detection coordinates from the detections object and adds to array for tracking
    for position in positionList:
        # detect --> [ymin, xmin, ymax, xmax]
        detect = detections['detection_boxes'][position]
        x, y, w, h = (
        detect[1] * camera_Width, detect[0] * camera_Height, detect[3] * camera_Width, detect[2] * camera_Height)
        listDetections.append([x, y, w, h])

    frame = cv.resize(image_np, frameSize)

    # Object Tracking
    boxes_ids = tracker.update(listDetections, frame, gps_controller, m)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv.putText(frame, str(id), (int(x), int(y) - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 3)

    if positionList != []:
        window[detectionKey].Widget.config(background='red')
        window[detectionKey2].Widget.config(background='red')
        # playsound._playsoundWin('alarm.wav')
        ### Moved - tracker.py calls det to write to log
        # display to textbox
        # for position in positionList:
        #     if (
        #     category_index.get(detections['detection_classes'][position] == detections['detection_classes'][position])):
        #         # print(str(category_index.get(detections['detection_classes'][position]+1)) + " on " + detectionKey)
        #         window["textbox"].update(window["textbox"].get() + "\n" + str(
        #             category_index.get(detections['detection_classes'][position] + 1)) + " on " + detectionKey)
        #         with open("Log.txt", "wt") as text_file:
        #             text_file.write(str(window["textbox"].get()))

        # print(str(detections['detection_classes']) + "was found on " + detectionKey)
    else:
        window[detectionKey].Widget.config(background='black')
        window[detectionKey2].Widget.config(background='black')

    # Update camera
    imgbytes = cv.imencode(".png", frame)[1].tobytes()
    window[detectionKey].update(data=imgbytes)


# Create folder for storing snapshots of detections (if not already created)
create_folder("detectionImages")
# Clear detections from past runs
delete_folder_contents("detectionImages")
#Clear log file
clearLog()

# Label map path
LABEL_MAP_NAME = 'Tensorflow/workspace/annotations/label_map.pbtxt'
#Model path
d2PathCkpt = 'Tensorflow/workspace/models/my_ssd_mobnet'
d2Config = 'Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config'
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(d2Config)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(d2PathCkpt, 'ckpt-51')).expect_partial()
# Label map formatted
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_NAME)
# Camera Settings
camera_Width  = 480 # 320 # 480 # 720 # 1080 # 1620
camera_Height = 360 # 240 # 360 # 540 # 810  # 1215
frameSize = (camera_Width, camera_Height)
video_capture1 = cv.VideoCapture('Video 2.mp4')
video_capture2 = cv.VideoCapture(1)
video_capture3 = cv.VideoCapture(2)
video_capture4 = cv.VideoCapture(3)
video_capture5 = cv.VideoCapture(4)
time.sleep(2.0)

# init Windows Manager
sg.theme("Black")

# def webcam col
cameracolumn_layout = [[sg.Text("Choose how many camera's you want to display", size=(60,1))], [sg.Combo(["1", "2", "3", "4", "5"], key="cameraAmount", default_value="1")]]

cameracolumn = sg.Column(cameracolumn_layout, element_justification='center', background_color="black")

mapbutton_layout = [[sg.Text("Open Map", size=(60,1), justification='center')], [sg.Button('Map')]]

mapbutton = sg.Column(mapbutton_layout, element_justification='center', background_color="black")

blankcolumn_layout = [[sg.Text("", size=(60,1))], [sg.Text("")]]

blankcolumn = sg.Column(blankcolumn_layout, element_justification='center', background_color="black")

threshcolumn_layout = [[sg.Text("Choose the threshold you want to use", size=(60,1))], [sg.Combo(["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], key='threshAmount', default_value="60%")]]

threshcolumn = sg.Column(threshcolumn_layout, element_justification='center', background_color="black")

colwebcam1_layout = [[sg.Text("Camera 1 (Front Driver)", size=(60, 1), background_color='black', justification="center")],
                        [sg.Image(filename="", key="cam1")]]
colwebcam1 = sg.Column(colwebcam1_layout, element_justification='center', key="cam1Update", background_color='black')

colwebcam2_layout = [[sg.Text("Camera 2 (Front Passenger)", size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="cam2")]]
colwebcam2 = sg.Column(colwebcam2_layout, element_justification='center', key="cam2Update", background_color='black')

colwebcam3_layout = [[sg.Text("Camera 3 (Driver Side)", size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="cam3")]]
colwebcam3 = sg.Column(colwebcam3_layout, element_justification='center', key="cam3Update", background_color='black')

colwebcam4_layout = [[sg.Text("Camera 4 (Passenger Side)", size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="cam4")]]
colwebcam4 = sg.Column(colwebcam4_layout, element_justification='center', key="cam4Update", background_color='black')

colwebcam5_layout = [[sg.Text("Camera 5 (Rear)", size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="cam5")]]
colwebcam5 = sg.Column(colwebcam5_layout, element_justification='center', key="cam5Update", background_color='black')

coltextbox_layout = [[sg.Text("Output", size=(60,1), justification="center")],
                        [sg.Multiline(size=(60, 30), key="textbox", autoscroll=True, disabled=True)]]
coltextbox = sg.Column(coltextbox_layout, element_justification='center')

colslayout = [[cameracolumn, mapbutton, threshcolumn], [colwebcam1, colwebcam2, colwebcam3], [colwebcam4, colwebcam5, coltextbox]]

layout = [colslayout]

window = sg.Window("FOD Detection", layout,
                    no_titlebar=False, alpha_channel=1, grab_anywhere=False,
                    return_keyboard_events=True, location=(100, 100)).Finalize()

# Initialize GPS controller
gps_controller = GPS_Controller()

#Initialize object tracking object
tracker = EuclideanDistTracker()

# Initialize coordinates
try: # if gps is accessible
    starting_coords = gps_controller.extract_coordinates()
except: # if unable to access gps
    starting_coords = [0,0]
    print("unable to get starting coordinates - defaulting to 0,0")

# Initialize map
m = createMap()

# Spawn thread for concurrent GPS reading (to bypass Input/Output delay of live reads)
thread = Thread(target=gps_controller.extract_coordinates)
thread.start()

while True:
    start_time = time.time()
    event, values = window.read(timeout=20)

    if not thread.is_alive():
        print("thread completed")
        thread = Thread(target=gps_controller.extract_coordinates)
        thread.start()

    if event == sg.WIN_CLOSED:
        break
    if event == 'Map':
        openMap()

    threshold = getThreshold(values['threshAmount'])
    cameraAmount = getCameraAmount(values['cameraAmount'])

    if cameraAmount == 1:
        ret, frame1 = video_capture1.read()
        tfBoundingBoxes(frame1, "cam1", "cam1Update", threshold)
    elif cameraAmount == 2:
        ret, frame1 = video_capture1.read()
        tfBoundingBoxes(frame1, "cam1", "cam1Update", threshold)
        ret, frame2 = video_capture2.read()
        tfBoundingBoxes(frame2, "cam2", "cam2Update", threshold)
    elif cameraAmount == 3:
        ret, frame1 = video_capture1.read()
        tfBoundingBoxes(frame1, "cam1", "cam1Update", threshold)
        ret, frame2 = video_capture2.read()
        tfBoundingBoxes(frame2, "cam2", "cam2Update", threshold)
        ret, frame3 = video_capture3.read()
        tfBoundingBoxes(frame3, "cam3", "cam3Update", threshold)
    elif cameraAmount == 4:
        ret, frame1 = video_capture1.read()
        tfBoundingBoxes(frame1, "cam1", "cam1Update", threshold)
        ret, frame2 = video_capture2.read()
        tfBoundingBoxes(frame2, "cam2", "cam2Update", threshold)
        ret, frame3 = video_capture3.read()
        tfBoundingBoxes(frame3, "cam3", "cam3Update", threshold)
        ret, frame4 = video_capture4.read()
        tfBoundingBoxes(frame4, "cam4", "cam4Update", threshold)
    elif cameraAmount == 5:
        ret, frame1 = video_capture1.read()
        tfBoundingBoxes(frame1, "cam1", "cam1Update", threshold)
        ret, frame2 = video_capture2.read()
        tfBoundingBoxes(frame2, "cam2", "cam2Update", threshold)
        ret, frame3 = video_capture3.read()
        tfBoundingBoxes(frame3, "cam3", "cam3Update", threshold)
        ret, frame4 = video_capture4.read()
        tfBoundingBoxes(frame4, "cam4", "cam4Update", threshold)
        ret, frame5 = video_capture5.read()
        tfBoundingBoxes(frame5, "cam5", "cam5Update", threshold)
        

video_capture1.release()
video_capture2.release()
video_capture3.release()
video_capture4.release()
video_capture5.release()
cv.destroyAllWindows()