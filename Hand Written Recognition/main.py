from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.metrics import CategoricalAccuracy
from keras.losses import categorical_crossentropy
from keras import callbacks
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import numpy as np

class Model:

    def build(self):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(2048, activation='leaky_relu'))
        model.add(Dense(1024, activation='leaky_relu'))
        model.add(Dense(512, activation='leaky_relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def load_data(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        return (x_train, y_train), (x_test, y_test)

    def normalize(self, x_train, x_test):
        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)

        return x_train, x_test


    def earlystopping(self):
        earlystopping = callbacks.EarlyStopping(monitor='loss',
                                                patience=5,
                                                verbose=1,
                                                mode='min',
                                                restore_best_weights=True)

        return earlystopping

    def fit_model(self, model, x_train, y_train, earlystopping):
        history = model.fit(x_train,
                            y_train,
                            batch_size=32,
                            verbose=1,
                            epochs=100,
                            callbacks=[earlystopping])

        return history

    def plot_training(self, history):
        plt.plot(history.history['loss'], color='b', label='Training Loss')
        plt.plot(history.history['val_loss'], color='r', label='Validation Loss') #burayı düzelt
        plt.legend()
        plt.show()

        plt.plot(history.history['categorical_accuracy'], color='b', label='Training  Accuracy')
        plt.plot(history.history['val_categorical_accuracy'], color='r', label='Validation Accuracy')
        plt.legend()
        plt.show()

    def pred(self,model, x_test):
        pred = model.predict(x_test)
        preds = []
        for i in pred:
            preds.append(np.argmax(i))
        preds = np.array(preds)
        return preds

    def confussionMatrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        display.plot()
        plt.show()
        print("Acc score : ", accuracy_score(y_test, y_pred))

    def save_model(self, model):
        model.save_weights('mnist.h5')
        with open('mnist.json', 'w') as f:
            f.write(model.to_json())


if __name__ == '__main__':
    model = Model()
    build_model = model.build()
    (x_train, y_train), (x_test, y_test) = model.load_data()
    (x_train, x_test) = model.normalize(x_train, x_test)
    earlystopping = model.earlystopping()
    history = model.fit_model(build_model, x_train, y_train, earlystopping)
    model.save_model(build_model)
    #model.plot_training(history)
    preds = model.pred(build_model, x_test)
    model.confussionMatrix(y_test, preds)
