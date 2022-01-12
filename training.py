import os
import object_detection
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

CUSTOM_MODEL_NAME = "my_ssd_mobnet"
PRETRAINED_MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
TF_RECORD_SCRIPT_NAME = "generate_tfrecord.py"
LABEL_MAP_NAME = "label_map.pbtxt"

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


##List the items that needs to be detect
labels = [{"name": "screw", "id": 1}]
##Create A label map
with open(files["LABELMAP"], "w") as f:
    for label in labels:
        f.write("item { \n")
        f.write("\tname:'{}'\n".format(label["name"]))
        f.write("\tid:{}\n".format(label["id"]))
        f.write("}\n")

## This parts updates config file according to labels list length and paths
print("Updating Config File")
config = config_util.get_configs_from_pipeline_file(files["PIPELINE_CONFIG"])
config

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files["PIPELINE_CONFIG"], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(
    paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME, "checkpoint", "ckpt-0"
)
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = files["LABELMAP"]
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
    os.path.join(paths["ANNOTATION_PATH"], "train.record")
]
pipeline_config.eval_input_reader[0].label_map_path = files["LABELMAP"]
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
    os.path.join(paths["ANNOTATION_PATH"], "test.record")
]

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files["PIPELINE_CONFIG"], "wb") as f:
    f.write(config_text)

print("Config File Updated")

first_command = "python {} -x {} -l {} -o {}".format(files['TF_RECORD_SCRIPT'], os.path.join(paths['IMAGE_PATH'], 'train'),
                         files['LABELMAP'], os.path.join(paths['ANNOTATION_PATH'], 'train.record'))
second_command = "python {} -x {} -l {} -o {}".format(files['TF_RECORD_SCRIPT'], os.path.join(paths['IMAGE_PATH'], 'test'),files['LABELMAP'], os.path.join(paths['ANNOTATION_PATH'], 'test.record'))

TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
training_command = "python {} --model_dir={} --pipeline_config_path={}--num_train_steps=2000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])

evaluate_command = "python {} --model_dir={} --pipeline_config_path={}--checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'],paths['CHECKPOINT_PATH'])

print("save these commands and run them")
print("1-) " + first_command)
print("2-) " + second_command)
print("3-) " + training_command)
print("4-) " + evaluate_command)
