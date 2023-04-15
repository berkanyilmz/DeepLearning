from keras import callbacks
from keras.applications import ConvNeXtBase, ConvNeXtTiny, ConvNeXtLarge, ConvNeXtSmall, ConvNeXtXLarge
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D
import keras
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.metrics import CategoricalAccuracy
import matplotlib.pyplot as plt
from keras.optimizers import Adam, Adamax, RMSprop
from sklearn import metrics

################### TRANSFER LEARNING ################
base_model = ConvNeXtSmall(
    weights='imagenet',
    input_shape=(48, 48, 3),
    include_top=False)
base_model.trainable = False

inputs = keras.Input(shape=(48, 48, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)

outputs = keras.layers.Dense(7, activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=[CategoricalAccuracy()])


# Reading dataset
data = ImageDataGenerator(rescale=1. / 255,
                          shear_range=0.2,
                          zoom_range=0.2,
                          horizontal_flip=True)

train_set = data.flow_from_directory('veriseti/train',
                                     target_size=(48, 48),
                                     #color_mode='grayscale',
                                     batch_size=32,
                                     class_mode='categorical',
                                     shuffle=False)

test_set = data.flow_from_directory('veriseti/test',
                                    target_size=(48, 48),
                                    #color_mode='grayscale',
                                    batch_size=32,
                                    class_mode='categorical',
                                    shuffle=False)

# Creating Early Stopping
earlystopping = callbacks.EarlyStopping(monitor='val_loss',
                                        mode='min',
                                        patience=7,
                                        restore_best_weights=True)

# Training the Model
history = model.fit(train_set,
          epochs=50,
          validation_data=test_set,
          validation_freq=1,
          callbacks=[earlystopping])

# Predict Value
test_set.reset()
pred = model.predict(test_set)
preds = []
for i in pred:
    preds.append(np.argmax(i))
preds = np.array(preds)



#karmaşıklık matrisi oluşturma
test_label = test_set.labels

cm = confusion_matrix(test_label, preds)
print(cm)
acc_score = accuracy_score(test_label, preds)
print('Accuracy Score : ', acc_score)


# Plotting Training Loss & Validation Loss
plt.plot(history.history['loss'], color='b', label='Training Loss')
plt.plot(history.history['val_loss'], color='r', label='Validation Loss')
plt.legend()
plt.show()
# Plotting the training accuracy and validation accuracy
plt.plot(history.history['categorical_accuracy'], color='b', label='Training  Accuracy')
plt.plot(history.history['val_categorical_accuracy'], color='r', label='Validation Accuracy')
plt.legend()
plt.show()

print('Validation Accuracy is ', np.max(history.history['val_categorical_accuracy']))

################################## FINE TUNING ##################################
base_model.trainable = True
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-5),
              metrics=[CategoricalAccuracy()])

history = model.fit(train_set,
               validation_data=test_set,
               epochs=50,
               callbacks=[earlystopping])

# Predict Value
test_set.reset()
pred = model.predict(test_set)
preds = []
for i in pred:
    preds.append(np.argmax(i))
preds = np.array(preds)

# Creating Confusion Matrix
test_label = test_set.labels

cm = confusion_matrix(test_label, preds)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
cm_display.plot()
plt.show()
acc_score = accuracy_score(test_label, preds)
print('Accuracy Score : ', acc_score)

model.save('ConvNeXtSmall')

# Plotting Training Loss & Validation Loss
plt.plot(history.history['loss'], color='b', label='Training Loss')
plt.plot(history.history['val_loss'], color='r', label='Validation Loss')
plt.legend()
plt.show()
# Plotting the training accuracy and validation accuracy
plt.plot(history.history['categorical_accuracy'], color='b', label='Training  Accuracy')
plt.plot(history.history['val_categorical_accuracy'], color='r', label='Validation Accuracy')
plt.legend()
plt.show()

print('Validation Accuracy is ', np.max(history.history['val_categorical_accuracy']))
