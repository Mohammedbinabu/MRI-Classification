import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten



#generator # same size for all the images
train_Dataset = keras.utils.image_dataset_from_directory(
    directory = r'/home/gpu-linux/Desktop/MRI-Classification/data/train',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size=(256,256)
) 

validation_Dataset = keras.utils.image_dataset_from_directory(
    directory = r'/home/gpu-linux/Desktop/MRI-Classification/data/validation',
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
model.add(Dense(4,activation='sigmoid'))

print(model.summary())

model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy']
              )
# Train the model
history = model.fit(train_ds,epochs=20,validation_data=validation_ds)

# Save in HDF5 format (produces a .h5 file)
model.save("my_model.h5")


