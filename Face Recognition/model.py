# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 17:15:44 2022

@author: berkan
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class Model:
    #this method build the model
    def __init__(self):
        self.classifier = Sequential()
        self.classifier.add(Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=3, strides=2))
        self.classifier.add(Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=3, strides=2))
        
        #self.classifier.add(Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'))
        #self.classifier.add(Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'))
        #self.classifier.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
        #self.classifier.add(MaxPooling2D(pool_size=3, strides=2))
        
        self.classifier.add(Flatten())
        #self.classifier.add(Dense(4096, activation='relu'))
        self.classifier.add(Dense(128, activation='relu'))
        
        self.classifier.add(Dense(1, activation='sigmoid'))
        self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print('Yapılandırıcı metod bitti')
        
        
    def readImages(self):
        train_datagen = ImageDataGenerator(rescale=1./255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)
        
        test_datagen = ImageDataGenerator(rescale=1./255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)
        
        self.training_set = train_datagen.flow_from_directory('data/Train',
                                                         target_size=(64,64),
                                                         batch_size=1,
                                                         class_mode='binary')
        
        self.test_set = test_datagen.flow_from_directory('data/Test',
                                                    target_size=(64,64),
                                                    batch_size=1,
                                                    class_mode='binary')
        
        
        
    def fitModel(self):
        self.classifier.fit_generator(self.training_set,
                                 steps_per_epoch=8000,
                                 epochs=1,
                                 validation_data=self.test_set,
                                 validation_steps=2000)
    
        
    
    def predict(self):
        self.test_set.reset()
        self.pred = self.classifier.predict(self.test_set)
        
        self.pred[self.pred > .5] = 1
        self.pred[self.pred <= .5] = 0
        return self.pred
        
    def labels(self):
        self.test_label = []
        for i in range(0,int(18167)):
            self.test_label.extend(np.array(self.test_set[i][1]))
        return self.test_label
        
    def createDataframe(self):
        fileName = self.test_set.filenames
        self.result = pd.DataFrame(columns=['File Names', 'Predicts', 'Test'])
        self.result['File Names'] = fileName
        self.result['Predicts'] = self.pred
        self.result['Test'] = self.test_label
        return self.result
        
        
    def confusionmatrix(self):
        cm = confusion_matrix(self.test_label, self.pred)
        print(cm)
        tn, fp, fn, tp = confusion_matrix(self.test_label, self.pred).ravel()
        total = tn + tp + fn + tp
        accuracyRate = (tp+tn) / total
        print ('Accuracy Rate : {}'.format(accuracyRate))
        
        
model = Model()
model.readImages()
model.fitModel()
prd = model.predict()
label = model.labels()
df = model.createDataframe()
model.confusionmatrix()
