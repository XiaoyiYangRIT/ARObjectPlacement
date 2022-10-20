# import the necessary packages
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
def create_mlp(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))
	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))
	# return our model
	return model
def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1
	# define the model input
	inputs = Input(shape=inputShape)
	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs
		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		# flatten the volume, then FC => RELU => BN => DROPOUT
		x = Flatten()(x)
		x = Dense(16)(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Dropout(0.5)(x)
		# apply another FC layer, this one to match the number of nodes
		# coming out of the MLP
		x = Dense(4)(x)
		x = Activation("relu")(x)
		# check to see if the regression node should be added
		if regress:
			x = Dense(1, activation="linear")(x)
		# construct the CNN
		model = Model(inputs, x)
		# return the CNN
		return model

class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()
        
        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual) 
        return out


class ResNet18(Model):

	def __init__(self, block_list, initial_filters=64):
		super(ResNet18, self).__init__()
		self.num_blocks = len(block_list) 
		self.block_list = block_list
		self.out_filters = initial_filters

		self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
		self.b1 = BatchNormalization()
		self.a1 = Activation('relu')
		self.blocks = tensorflow.keras.models.Sequential()

		for block_id in range(len(block_list)): 
			for layer_id in range(block_list[block_id]):

				if block_id != 0 and layer_id == 0:  
					block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
				else: 
					block = ResnetBlock(self.out_filters, residual_path=False)
				self.blocks.add(block) 
			self.out_filters *= 2 





	def call(self, width, height, depth):
		inputShape = (height, width, depth)
		chanDim = -1
		# define the model input
		inputs = Input(shape=inputShape)
		x = self.c1(inputs)
		x = self.b1(x)
		x = self.a1(x)
		x = self.blocks(x)
		# change here into dense layer and add flatten
		x = Flatten()(x)
		y = Dense(4)(x)
		#x = self.p1(x)
		#y = self.f1(x)

		# self added layer
		y = Activation("relu")(y)
		
		model = Model(inputs, y)
		# return y
		return model