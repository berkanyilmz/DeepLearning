from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, BatchNormalization, Activation, Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#dataset = https://www.kaggle.com/datasets/chiragsoni/ferdata

class Model:
    # this method build the model
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
        #self.model.add(BatchNormalization())
        self.model.add(Activation('leaky_relu'))
        self.model.add(MaxPooling2D((2,2)))

        self.model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('leaky_relu'))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        self.model.add(Activation('leaky_relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))

        self.model.add(Flatten())
        self.model.add(Activation('leaky_relu'))
        self.model.add(Dense(128))
        self.model.add(Dropout(0.5))
        self.model.add(Activation('leaky_relu'))
        self.model.add(Dense(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128))
        self.model.add(BatchNormalization())
        #self.model.add(Activation('relu'))
        self.model.add(Dense(7, activation='softmax'))

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['acc'])


    def readImages(self):
        print('görüntü okuma')
        data = ImageDataGenerator(rescale=1. / 255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

        train_set = data.flow_from_directory('train',
                                             target_size=(48, 48),
                                             batch_size=32,
                                             class_mode='categorical')

        test_set = data.flow_from_directory('test',
                                            target_size=(48, 48),
                                            batch_size=32,
                                            class_mode='categorical')
        return train_set, test_set

    def fitModel(self, train, test):
        print('eğitim')

        history = self.model.fit(train,
                                 epochs=20,
                                 validation_data=test,
                                 validation_freq=1)
        return history

    def predict(self, test):
        print('tahmin')
        test.reset()
        pred = self.model.predict(test)

        return pred

    def labels(self, test):
        print('etiket')
        test_label = []
        for i in test.labels:
            test_label.append(i)

        return test_label

    def confusionmatrix(self, test_label, pred):
        print('matrix')
        cm = confusion_matrix(test_label, pred)
        print(cm)
        tn, fp, fn, tp = confusion_matrix(test_label, pred).ravel()
        total = tn + tp + fn + tp
        accuracyRate = (tp + tn) / total
        print('Accuracy Rate : {}'.format(accuracyRate))

    def visualize(self, history):
        f, ax = plt.subplots(2, 1)

        # Assigning the first subplot to graph training loss and validation loss
        ax[0].plot(history.history['loss'], color='b', label='Training Loss')
        ax[0].plot(history.history['val_loss'], color='r', label='Validation Loss')

        # Plotting the training accuracy and validation accuracy
        ax[1].plot(history.history['acc'], color='b', label='Training  Accuracy')
        ax[1].plot(history.history['val_acc'], color='r', label='Validation Accuracy')

        plt.legend()
        plt.show()
        print('Accuracy Score = ', np.max(history.history['val_acc']))


model = Model()
train_data, test_data = model.readImages()
labels = test_data.labels
history = model.fitModel(train_data, test_data)
print(history.history.keys())
pred = model.predict(test_data)
print(pred)
label = model.labels(test_data)
model.visualize(history)