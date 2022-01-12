import os
import random
import shutil


# Edit these paths before using the scripts
train_folder_path =  'D:/Python/Projects/splitData/LuggageTagPhotos&Annotations(11-18)/train'
test_folder_path =   'D:/Python/Projects/splitData/LuggageTagPhotos&Annotations(11-18)/test'
target_folder_path = 'D:/Python/Projects/splitData/LuggageTagPhotos&Annotations(11-18)/target'
source_folder_path = 'D:/Python/Projects/splitData/LuggageTagPhotos&Annotations(11-18)'

train_folder = []
test_folder = []
target_folder = []
count = 0
# For windows use '\\' instead of '/'
for path, dir, files in os.walk(source_folder_path):
    if count == 1:
        break;
    for file in files:
        if file[-1] == 'g':
            print(file)
            value = random.randint(1, 10)
            if value > 7:
                shutil.copy(source_folder_path + '/' + file, test_folder_path + '/' + file)
                test_folder.append(file[:-3])
            elif 1 < value < 8:
                shutil.copy(source_folder_path + '/' + file, train_folder_path + '/' + file)
                train_folder.append(file[:-3])
            elif value < 2:
                shutil.copy(source_folder_path + '/' + file, target_folder_path + '/' + file)
                target_folder.append(file[:-3])
    count += 1


for file in train_folder:
    shutil.copy(source_folder_path + '/' + file + 'xml', train_folder_path + '/' + file + 'xml')

for file in test_folder:
    shutil.copy(source_folder_path + '/' + file + 'xml', test_folder_path + '/' + file + 'xml')

for file in target_folder:
    shutil.copy(source_folder_path + '/' + file + 'xml', target_folder_path + '/' + file + 'xml')
