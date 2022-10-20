import os

#path to the folder of screenshots
image_save_path = '/Users/anony/Downloads/Apple_Data_distance_estimation-master/FullDataMap_231/'
changePath = image_save_path + 'train/'
#change the name of all chair screenshots
for index in range(218, 470):
    os.rename(changePath + str(index) + '_chair.png', changePath + str(int(index)-217) + '_chair.png')
#change the name of all lamp screenshots
for index in range(470, 721):
    os.rename(changePath + str(index) + '_lamp.png', changePath + str(int(index)-217) + '_lamp.png')
