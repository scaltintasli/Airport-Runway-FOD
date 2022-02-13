# Airport-Runway-FOD

Our software intends to do it with the help of a high definition camera and machine learning capabilities. It will be able to detect foreigh object debris on runways quickly and efficiently giving airport staff time to clean up runways and most importantly keep our travel plans on schedule. 

In order to complete the task mentioned above there are a series of goals and objectives our software must include to make this technology into a reality. 

To start, the goals we have identified as most important for this software to achieve or be capable of doing are as follows:

* The ability to detect FOD on the runway including:
    * Bolts, screws, tools, people, animals, and any other large debris
* The implementation of a machine learning model to recognize FODs
* The software useability to be friendly enough to be used by airport personnel 

Along with these goals we have also identified the objectives that would be most important to keep track of throughout the development lifecycle of this software. These include:

* Acquiring domain knowledge by talking to experts regarding FOD
* Using machine learning to detect and recognize the FOD
* Use phones/cameras/drones/datasets to train a model to detect common types of FOD
* Use of notifications notifying airport personnel when FOD is located on a runway/tarmac

## Requirements
* Python 3.8
* Protoc
* Tensorflow Object Detection API

if you want to use your GPU:
* Cuda 11.2
* CUDDNN 8.2

## Installation


```git clone https://github.com/scaltintasli/Airport-Runway-FOD.git```

```git clone https://github.com/tensorflow/models```

```git clone https://github.com/nicknochnack/GenerateTFRecord```

```pip install -r requirements.txt```

```cd Tensorflow\models\research && protoc object_detection\protos\*.proto --python_out=. && copy object_detection\packages\tf2\setup.py setup.py && python setup.py build && python setup.py install```

```cd Tensorflow/models/research/slim && pip install -e .```


## Usage

TRANING

Edit label list and add items to list(these label names should match with images labels and it is case sensetive). the list is there :https://github.com/scaltintasli/Airport-Runway-FOD/blob/main/train_image.py#L48

Add images to test folder and train folder(with the .xml files)
run ```python train_image.py```
 
DETECTION

Edit load_train_model.py and make sure checkpoint number matches(Tensorflow\workspace\models\modelname)

run ```python load_train_model.py```

For live detection run ```python live_detection_GUI```
For Manual test run ```python detect_image.py```
