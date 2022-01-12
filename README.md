# Airport-Runway-FOD

Our software intends to do it with the help of a high definition camera and machine learning capabilities. It will be able to detect foreigh object debris on runways quickly and efficiently giving airport staff time to clean up runways and most importantly keep our travel plans on schedule. 

In order to complete the task mentioned above there are a series of goals and objectives our software must include to make this technology into a reality. 

To start, the goals we have identified as most important for this software to achieve or be capable of doing are as follows:

Markup : *The ability to detect FOD on the runway including:
    *Bolts, screws, tools, people, animals, and any other large debris
*The implementation of a machine learning model to recognize FODs
*The software useability to be friendly enough to be used by airport personnel 
Along with these goals we have also identified the objectives that would be most important to keep track of throughout the development lifecycle of this software. These include:
*Acquiring domain knowledge by talking to experts regarding FOD
*Using machine learning to detect and recognize the FOD
*Use phones/cameras/drones/datasets to train a model to detect common types of FOD
*Use of notifications notifying airport personnel when FOD is located on a runway/tarmac

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install TDB
```

## Usage

```python
import TBD

# returns 'words'
TBD.pluralize('word')

# returns 'geese'
TBD.pluralize('goose')

# returns 'phenomenon'
TBD.singularize('phenomena')
```
1-) Clone this repo
    
    THEN Go to Tensorflow and clone this https://github.com/tensorflow/models
    
    And finally go to Tensorflow scripts and clone this https://github.com/nicknochnack/GenerateTFRecord

2-) Run ‘pipenv install’

3-) Run ‘pipenv shell’ then create a new branch git checkout -b "branch-name" then run these two:
   
   `cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install`
    
   `cd Tensorflow/models/research/slim && pip install -e .`


4-) Edit label list and add items to list(these label names should match with images labels and it is case sensetive). the list is there :https://github.com/scaltintasli/Airport-Runway-FOD/blob/main/train_image.py#L48

5-) Add images to test folder and train folder(with the .xml files) 

6-) run ‘python train_image.py”

7-) run ‘printed command one by one’

8-) first two commands should be really fast

9-) third command for the training so it might take a while to train

10-) This will evaluate the model you need to quit the process in the end with ctrl + C

11-) Edit load_train_model.py and make sure checkpoint number matches

12-) Run ‘python load_train_model.py’

13-) Edit detect_image.py and make sure the image path matches with your image

14-) Then run detect_image.py it should generate a image file in the main folder
