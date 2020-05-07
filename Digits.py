import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import scipy.misc
import PIL
import imageio
from tensorflow.keras.preprocessing import image

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Preprocessing of data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Building the model LeNet-5
model = models.Sequential()
model.add(layers.Conv2D(6, (5, 5), strides = (1,1), padding = 'valid', activation='relu', kernel_initializer='glorot_uniform', 
	                    input_shape=(28, 28, 1)))
model.add(layers.AveragePooling2D(pool_size=(2, 2), padding = 'valid'))
model.add(layers.Conv2D(16, (5, 5), strides = (1,1), padding = 'valid', activation='relu', kernel_initializer='glorot_uniform',))
model.add(layers.AveragePooling2D(pool_size=(2, 2), padding = 'valid'))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation = 'relu', kernel_initializer='glorot_uniform'))
model.add(layers.Dense(84, activation = 'relu', kernel_initializer='glorot_uniform'))
model.add(layers.Dense(10, activation = 'softmax'))

adam = Adam(learning_rate = 0.01 , beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, name = 'Adam')
model.compile(optimizer = adam, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 16, epochs = 2)

preds = model.evaluate(x_test, y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

model.summary()

q = input('Please input the number of images you want to test\n')
w = int(q)
for t in range(w) :
	img_path = input('Please enter the name of image\n')
	img = image.load_img(img_path, target_size=(28, 28))
	img = img.convert(mode='L')
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = x/255.0
	my_image = imageio.imread(img_path)
	pred = model.predict(x)
	print(pred)
	print('The number written in paper is:',np.argmax(pred))

print("\n")
print("***********************************************************************************************************")
print("THANK YOU FOR AVAILING THIS SERVICE")
print("This CNN had been implemented by AVIRAL SINGH")
print("***********************************************************************************************************")

