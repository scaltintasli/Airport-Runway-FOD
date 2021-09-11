# Airport-Runway-FOD

1-) Clone the repo

2-) Run ‘pipenv install’

3-) Run ‘pipenv shell’

4-) Edit label list and add items to list(these label names should match with images labels and it is case sensetive)

5-) Add images to test folder and train folder(with the .xml files) 6-) run ‘python train_image.py”

7-) run ‘printed command one by one’

8-) first two commands should be really fast

9-) third command for the training so it might take a while to train

10-) This will evaluate the model you need to quit the process in the end with ctrl + C

11-) Edit load_train_model.py and make sure checkpoint number matches

12-) Run ‘python load_train_model.py’

13-) Edit detect_image.py and make sure the image path matches with your image

14-) Then run detect_image.py it should generate a image file in the main folder
