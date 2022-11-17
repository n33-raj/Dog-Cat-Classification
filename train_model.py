## Importing Libraries

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np







## Preprocessing the training data
train_datagen = ImageDataGenerator(rescale = 1./255,               ## data agumentation
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('Dataset/training_set',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = "binary")



## Preprocessing the test data

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('Dataset/test_set',
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode = "binary")



### Building CNN

cnn = models.Sequential()                  ## initialising cnn

## adding convolution layer
cnn.add(layers.Conv2D(filters=32,                
                              kernel_size=3,
                              activation='relu',
                              input_shape=[64,64,3]))

## adding pooling
cnn.add(layers.MaxPool2D(pool_size=2,strides=2))      


## adding second convolution layer
cnn.add(layers.Conv2D(filters=64,
                              kernel_size=3,
                              activation='relu'))

## adding pooling
cnn.add(layers.MaxPool2D(pool_size=2,strides=2))

## adding third convolution layer
cnn.add(layers.Conv2D(filters=64,
                              kernel_size=3,
                              activation='relu'))

## adding pooling
cnn.add(layers.MaxPool2D(pool_size=2,strides=2))

## Flattening
cnn.add(layers.Flatten())


## Full Connection
cnn.add(layers.Dense(units=128, activation='relu'))

## Output Layer
cnn.add(layers.Dense(units=1, activation='sigmoid'))



### Training The CNN
cnn.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])     ## compiling the CNN


### Training the CNN on the Training set and evaluating it on the Test set
H = cnn.fit(training_set, validation_data = test_set,  epochs = 25)


### Model Summary
print(cnn.summary())


## Saving Model
cnn.save('dog_cat.h5')


## plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = 25
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

## save plot to disk
plt.savefig('plot.png')