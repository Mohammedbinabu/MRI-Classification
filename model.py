import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten


#generator # same size for all the images
train_Dataset = keras.utils.image_dataset_from_directory(
    directory = 'C:/Users/ADMIN/Desktop/ibr/alzheimer_dataset/train',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size=(256,256)
) 

validation_Dataset = keras.utils.image_dataset_from_directory(
    directory = 'C:/Users/ADMIN/Desktop/ibr/alzheimer_dataset/test',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size=(256,256)
) 

def normalize(img,label):
    image = tf.cast(img/255,tf.float32)
    return image,label

train_ds = train_Dataset.map(normalize)
print('\nDone with training data normalization (1,0)')
validation_ds = validation_Dataset.map(normalize)
print('\nDone with validation data normalization (1,0)')

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

print(model.summary())

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_ds,epochs=10,validation_data=validation_ds)