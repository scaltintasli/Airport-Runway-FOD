import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Always check custom model name and update it if it is not correct
# And Check label_map.pbtxt too, make sure it has right object labels
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
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


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
