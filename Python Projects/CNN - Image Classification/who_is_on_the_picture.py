import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


W_grid = 15
L_grid = 15
fig, axes = plt.subplots(L_grid, W_grid, figsize=(25, 25))
axes = axes.ravel() 

n_training = len(X_train)

for i in np.arange(0, L_grid * W_grid):
    index = np.random.randint(0, n_training) 
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])
    axes[i].axis('off')

plt.subplots(hspace=0.4)


import keras 
no_c = 10
y_train = keras.utils.to_categorical(y_train, no_c)
y_test = keras.utils.to_categorical(y_test, no_c)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255 
X_test = X_test/255

Input_shape = X_train.shape[1:]

# TRAIN
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

m = Sequential()

m.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=Input_shape))
m.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
m.add(MaxPooling2D(2,2))
m.add(Dropout(0.3))

m.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
m.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
m.add(MaxPooling2D(2,2))
m.add(Dropout(0.2))

m.add(Flatten())
m.add(Dense(units=512, activation='relu'))
m.add(Dense(units=512, activation='relu'))
m.add(Dense(units=10, activation='softmax')) 

m.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

history = m.fit(X_train, y_train, batch_size=32, epochs=2, shuffle=True)

evaluation = m.evaluate(X_test, y_test)
print('Test Accuracy: {}'.format(evaluation[1]))

predicted_classes = m.predict_classes(X_test)
print(predicted_classes)

y_test = y_test.argmax(1)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
print(cm)

import seaborn as sns
plt.figure(figsize = (10,10))
sns.heatmap(cm, annot=True)



# IMAGE AUGUMENTATION
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator (
            rotation_range = 90,
            width_shift_range = 0.1,
            horizontal_flip = True,
            vertical_flip = True                
        )

datagen.fit(X_train)
m.fit_generator(datagen.flow(X_train, y_train, batch_size=32), epochs=1)

score = m.evaluate(X_test, y_test)
print('Test accuracy', score[1])


