# This file saves 3 rotated versions of every image in the folder.
# NOTE: Only run this once. Additional runs within the same folder will results in exponential duplicates.
# INSTRUCTIONS: Put this script in the folder of the input images, and then run the script once.
#  This will save 3 images per input image, each rotated another 90 degrees.

from PIL import Image
import os

input_directory_path = os.getcwd()
output_directory_path = os.path.join(input_directory_path, "rotated")
print(input_directory_path)
print(output_directory_path)

for entry in os.scandir(input_directory_path):
    if (entry.path.endswith(".jpg")
            or entry.path.endswith(".png")) and entry.is_file():
        print(entry.path)
        image_file = entry.path
        image = Image.open(image_file)

        # save a 90 rotation
        image = image.rotate(90, expand=True)
        image.save(image_file[0:-4] + "r90.jpg")

        # save a 180 rotation
        image = image.rotate(90, expand=True)
        image.save(image_file[0:-4] + "r180.jpg")

        # save a 270 rotation
        image = image.rotate(90, expand=True)
        image.save(image_file[0:-4] + "r270.jpg")
        
