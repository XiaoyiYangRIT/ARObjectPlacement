# import the necessary packages
from model import datasets
from model import models
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
import numpy as np
import argparse
import locale
import os
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of apple images")
args = vars(ap.parse_args())# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading apple attributes...")
inputPath = 'Apple_dataset_info.txt'

# full data set, three categories,  five features
inputPathTrain = 'Full_dataset_info_train.txt'
inputPathTest = 'Full_dataset_info_test.txt'
# df = datasets.load_house_attributes(inputPath)


# #1_ three categories, four feature, removed categorical data
# inputPathTrain = '1_noCategorical_train.txt'
# inputPathTest = '1_noCategorical_test.txt'


# #2_ crossObject_noCategorical_123, four feature
# inputPathTrain = '2_crossObject_noCategorical_123_train.txt'
# inputPathTest = '2_crossObject_noCategorical_123_test.txt'


# #2_ crossObject_noCategorical_132, four feature
# inputPathTrain = '2_crossObject_noCategorical_132_train.txt'
# inputPathTest = '2_crossObject_noCategorical_132_test.txt'


# #2_ crossObject_noCategorical_231, four feature
# inputPathTrain = '2_crossObject_noCategorical_231_train.txt'
# inputPathTest = '2_crossObject_noCategorical_231_test.txt'


df_train = datasets.load_house_attributes(inputPathTrain)
df_test = datasets.load_house_attributes(inputPathTest)
print('df_train:')
print(df_train)
print('df_test:')
print(df_test)

# load the apple images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading apple images...")
# images = datasets.load_house_images(df, args["dataset"])
print(args["dataset"] + '/train')
images_train = datasets.load_fulldataset_train_images(df_train, args["dataset"] + '/train')
images_test = datasets.load_fulldataset_test_images(df_test, args["dataset"] + '/test')
images_train = images_train / 255.0
images_test = images_test / 255.0

# images = images / 255.0
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")
# split = train_test_split(df, images, test_size=0.25, random_state=42)
# (trainAttrX, testAttrX, trainImagesX, testImagesX) = split
images_train, df_train = sklearn.utils.shuffle(images_train, df_train)
# images_test, df_test = sklearn.utils.shuffle(images_test, df_test)

(trainAttrX, testAttrX, trainImagesX, testImagesX) = df_train, df_test, images_train, images_test

print('trainAttrX', trainAttrX.shape) # (sample num x feature num)
print('testAttrX', testAttrX.shape) # (sample num x feature num)
print('trainImagesX', trainImagesX.shape) # (sample num x imgs(3d))
print('testImagesX', testImagesX.shape) # (sample num x imgs(3d))

# find the largest house price in the training set and use it to
# scale our house prices to the range [0, 1] (will lead to better
# training and convergence)
maxPrice = 1
# maxPrice = trainAttrX["GT_label"].max()

trainY = trainAttrX["GT_label"] / maxPrice
# trainY = np.tile(trainY, 10)
# print(trainY.shape)
testY = testAttrX["GT_label"] / maxPrice
# process the apple attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together
# (trainAttrX, testAttrX) = datasets.process_house_attributes(df,
# 	trainAttrX, testAttrX)

(trainAttrX, testAttrX) = datasets.process_house_attributes(trainAttrX, testAttrX)
# create the MLP and CNN models
mlp = models.create_mlp(trainAttrX.shape[1], regress=False)


# cnn = models.ResNet18([2, 2, 2, 2])
# cnn = cnn.call(128, 128, 3)
cnn = models.create_cnn(128, 128, 3, regress=False)

# print("mlp")
# print(mlp.shape)
# print(mlp.output.shape)

# print('cnn.output.shape', cnn.output.shape) # (None x 4) for origin place
# print("cnn")
# print(cnn.shape)
# print(cnn.output.shape)
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])
# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="sigmoid")(x)
print(x.shape)
# exit()

# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted realistic value)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)
# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual realistic value*
opt = Adam(lr=1e-4, decay=1e-3 / 200) #1e-4 origin
# opt =SGD(lr=0.1)
model.compile(loss="mean_squared_error", optimizer=opt)
# train the model
print("[INFO] training model...")
# print(trainAttrX.shape) # (162, 6)
# print(trainImagesX.shape) # (162, 64, 64, 3)
# trainAttrX = np.tile(trainAttrX, (10,1))
# trainImagesX = np.tile(trainImagesX, (10, 1, 1, 1))
# print(trainAttrX.shape) # (162, 6)
# print(trainImagesX.shape) # (162, 64, 64, 3)
model.fit(
	x=[trainAttrX, trainImagesX], y = trainY,
	validation_data=([testAttrX, testImagesX], testY),
	epochs=2000, batch_size=8)
# make predictions on the testing data
print("[INFO] predicting apple position...")
preds = model.predict([testAttrX, testImagesX])
# compute the difference between the *predicted* apple distance and the
# *actual* apple distance, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
print('pred_value',preds)
# print('testY', testY)
diff_round = np.round(preds.flatten(), 1) - testY
# Ceil_round = np.ceil(preds.flatten(), 1) - testY
# Floor_round = np.floor(preds.flatten(), 1) - testY

preds_memo = np.round(preds.flatten(), 1)
testY = testY.tolist()

print("preds_memo:")
print(preds_memo)

print("testY:")
print(testY)
# print(preds_memo==testY)
# print(np.count_nonzero(int(10*preds_memo)==int(10*testY)))
# print(len(testY))
# DiffMatchACC_1 = np.count_nonzero(int(preds_memo*10) == int(testY*10))/len(testY)
# print('DiffMatchACC_1', DiffMatchACC_1)
count = 0
for i in range(len(preds_memo)):
	if int(preds_memo[i]*10) == int(testY[i]*10):
		count +=1
		# print('yes')
DiffMatchACC = count/len(testY)

countRound = 0
for i in range(len(preds_memo)):
	if int(preds_memo[i]*10) == int(testY[i]*10) or int(preds_memo[i]*10 + 1) == int(testY[i]*10) or int(preds_memo[i]*10 - 1) == int(testY[i]*10):
		countRound +=1
		# print('yes')
DiffMatchACC_ROUND = countRound/len(testY)
# CeilMatchACC = np.count_nonzero(np.ceil(preds.flatten(), 1) == testY)/len(testY)
# FloorMatchACC = np.count_nonzero(np.floor(preds.flatten(), 1) == testY)/len(testY)

percentDiff = (diff / testY) * 100
percentDiffRound = (diff_round / testY) * 100
# percentCeilRound = (Ceil_round / testY) * 100
# percentFloorRound = (Floor_round / testY) * 100

absPercentDiff = np.abs(percentDiff)
absPercentDiffRound = np.abs(percentDiffRound)
# absPercentCeilRound = np.abs(percentCeilRound)
# absPercentFloorRound = np.abs(percentFloorRound)
# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
meanRound = np.mean(absPercentDiffRound)
stdRound = np.std(absPercentDiffRound)
# meanCeilRound = np.mean(absPercentCeilRound)
# stdCeilRound = np.std(absPercentCeilRound)
# meanFloorRound = np.mean(absPercentFloorRound)
# stdFloorRound = np.std(absPercentFloorRound)



# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
# print("[INFO] avg. GT_label: {}, std GT_label: {}".format(
# 	locale.currency(df["GT_label"].mean(), grouping=True),
# 	locale.currency(df["GT_label"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
print("[INFO] meanRound: {:.2f}%, stdRound: {:.2f}%".format(meanRound, stdRound))
# print("[INFO] meanCeilRound: {:.2f}%, stdCeilRound: {:.2f}%".format(meanCeilRound, stdCeilRound))
# print("[INFO] meanFloorRound: {:.2f}%, stdFloorRound: {:.2f}%".format(meanFloorRound, stdFloorRound))
# print("[INFO] DiffMatchACC: {:.2f}%, CeilMatchACC: {:.2f}%, FloorMatchACC: {:.2f}%".format(DiffMatchACC, CeilMatchACC, FloorMatchACC))
print("[INFO] DiffMatchACC: {:.2f}%".format(DiffMatchACC*100))
print("[INFO] DiffMatchACC: {:.2f}%".format(DiffMatchACC_ROUND*100))