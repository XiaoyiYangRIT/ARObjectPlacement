import os
import shutil

#Parameters
#path to the full data set of screenshots
DatasetPath = '/Users/anony/Downloads/ARTestOracle/FullData/'
#path to the training set of apple screenshots
GT_Path_train_apple = DatasetPath + 'train/applelabels'
#path to the training set of chair screenshots
GT_Path_train_chair = DatasetPath + 'train/chairlabels'
#path to the training set of lamp screenshots
GT_Path_train_lamp = DatasetPath + 'train/lamplabels'
#path to the testing set of screenshots
GT_Path_test = DatasetPath + 'test/testFinalLabels'

#Apple data set to extract naming conventions
image_save_path = '/Users/anony/Downloads/Apple_Data_distance_estimation-master/FullDataMap/'

#Textual screenshot name list of training set 
TxtPath = '/Users/anony/Downloads/Apple_Data_distance_estimation-master/Full_dataset_info_train.txt'

#Textual screenshot name list of testing set 
TxtPath_2 = '/Users/anony/Downloads/Apple_Data_distance_estimation-master/Full_dataset_info_test.txt'

#List names of all screenshots
Lines = []
for GT_Path in [GT_Path_train_apple, GT_Path_train_chair, GT_Path_train_lamp]:
    file = open(GT_Path, 'r')
    Lines.append(file.readlines())
# append into single long list
Lines_Memo_train = []
for i in range(len(Lines)):
    for item in Lines[i]:
        Lines_Memo_train.append(item)
print('train imgs num:', len(Lines_Memo_train))

Lines_test = []
for GT_Path in [GT_Path_test]:
    file = open(GT_Path, 'r')
    Lines_test.append(file.readlines())
# append into single long list
Lines_Memo_test = []
for i in range(len(Lines_test)):
    for item in Lines_test[i]:
        Lines_Memo_test.append(item)
print('test imgs num:', len(Lines_Memo_test))

num = 1
class_names= ['apple', 'chair', 'lamp']
for class_name in class_names:
    for filename in os.listdir(DatasetPath + 'train/' + class_name):
        for line in Lines_Memo_train:
            filename_fromGT = line.split('/')[5].split(' ')[0].split('\t')[0]
            GT_value = line.split('/')[5].split(' ')[0].split('\t')[1][0:3]
            # extract feature values from the screenshot names for the training set 
            if filename_fromGT == filename:
                print('yes')
                position = filename.split('_')[2]
                class_name = filename.split('_')[1]
                CO_distance = filename.split('_')[3]
                if CO_distance[0] == '.':
                    CO_distance = '0' + CO_distance
                Hori = filename.split('_')[4]
                if Hori[0] == '.':
                    Hori = '0' + Hori
                Verti = filename.split('_')[5].split('-')[0]
                if Verti[0] == '.':
                    Verti = '0' + Verti
                position = str(position)
                CO_distance = str(CO_distance)
                Hori = str(Hori)
                Verti = str(Verti)
                new_name= str(num) + '_' + class_name + '.png'
                class_name_str = class_name
                if class_name == 'apple':
                    class_name = 1
                elif class_name == 'chair':
                    class_name = 2
                elif class_name == 'lamp':
                    class_name = 3
                class_name = str(class_name)
                GT_value = str(GT_value)
                # STEP 1: copy and create the txt file

                newfile_path=os.path.join(image_save_path, 'train/', new_name)
                print(DatasetPath + filename_fromGT)
                print(newfile_path)
                shutil.copyfile(DatasetPath + 'train/' + class_name_str + '/' + filename_fromGT, newfile_path)
                with open(TxtPath, 'a') as f:
                    f.write(position + ' ' + class_name + ' ' + CO_distance + ' ' + Hori + ' ' + Verti + ' ' + GT_value)
                    # f.write(position + class_name + CO_distance +  Hori + Verti)
                    f.write('\r\n')
                print(position, class_name, CO_distance, Hori, Verti, GT_value)
                num +=1

num = 1
class_names_test = ['apple', 'chair', 'lamp']
for class_name in class_names_test:
    for filename in os.listdir(DatasetPath + 'test/' + class_name):
        for line in Lines_Memo_test:
            print(line)
            filename_fromGT = line.split('/')[5].split(' ')[0].split('\t')[0]
            # extract feature values from the screenshot names for the testing set 
            GT_value = line.split('/')[5].split(' ')[0].split('\t')[1][0:3]
            if filename_fromGT == filename:
                print('yes')
                position = filename.split('_')[5]
                class_name = filename.split('_')[4]
                CO_distan
				ce = filename.split('_')[6]
                if CO_distance[0] == '.':
                    CO_distance = '0' + CO_distance
                Hori = filename.split('_')[7]
                if Hori[0] == '.':
                    Hori = '0' + Hori
                Verti = filename.split('_')[8].split('-')[0]
                if Verti[0] == '.':
                    Verti = '0' + Verti
                position = str(position)
                CO_distance = str(CO_distance)
                Hori = str(Hori)
                Verti = str(Verti)
                new_name= str(num) + '_' + class_name + '.png'
                class_name_str = class_name
                if class_name == 'apple':
                    class_name = 1
                elif class_name == 'chair':
                    class_name = 2
                elif class_name == 'lamp':
                    class_name = 3
                class_name = str(class_name)
                GT_value = str(GT_value)
                print('GT_value', GT_value)
                newfile_path=os.path.join(image_save_path, 'test/', new_name)
                shutil.copyfile(DatasetPath + 'test/' + class_name_str + '/' + filename_fromGT, newfile_path)
                with open(TxtPath_2, 'a') as f:
                    f.write(position + ' ' + class_name + ' ' + CO_distance + ' ' + Hori + ' ' + Verti + ' ' + GT_value)
                    f.write('\r\n')
                print(position, class_name, CO_distance, Hori, Verti, GT_value)
                num +=1

    