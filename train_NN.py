import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

import app

input_data, labels = load_galaxy_data()
print(f' Input Data : {input_data.shape} \n Labels: {labels.shape}') #i/p: (# img in batch,pixel_length, pixel_width,#channels)
#o/p:(#img in batch, #labels)

#train-test dataset
X_train, X_valid, Y_train, Y_valid= train_test_split(input_data,labels,test_size=0.20,random_state=222,shuffle=True,stratify=labels)

#preprocess i/p
data_generator = ImageDataGenerator(rescale=1./255) #ImageDataGenerator object will normalize the pixels

#training_iterator
training_iterator=data_generator.flow(X_train,Y_train, batch_size=5)
#validation_iterator
validation_iterator=data_generator.flow(X_valid,Y_valid, batch_size=5)
#.flow: generates batches of data

#create model:
model=tf.keras.Sequential()
#Input: shape(pixel_l,pixel_w,channel#)
model.add(tf.keras.Input(shape=(128,128,3)))
#conv-1st layer
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) #Conv2D(#filters, size, strides) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=(2,2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Flatten())
#output: 4-features -> 4-classes
model.add(tf.keras.layers.Dense(16,activation="relu"))

model.add(tf.keras.layers.Dense(4,activation="softmax"))
#compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])
#model summary
print(model.summary())
#train model:
model.fit(
        training_iterator,
        steps_per_epoch=len(X_train)/5,
        epochs=8,
        validation_data=validation_iterator,
        validation_steps=len(X_valid)/5)


