import cv2 as cv
from keras import models
import numpy as np

def load_model():
    model = models.Sequential()

    with open('face.json', 'r') as f:
        model = models.model_from_json(f.read())

    model.load_weights('face_recognition.h5')

    return model

def face_detect():
    cap = cv.VideoCapture(0)
    model = load_model()
    text = ''

    while True:
        ret, frame = cap.read()

        rgbImage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgbImage = cv.resize(rgbImage, (48,48))
        rgbImage = np.reshape(rgbImage, (1,48,48,3))

        pred = model.predict(rgbImage)
        #pred = np.argmax(pred)

        if pred > .5:
            text = 'not face'
        else:
            text = 'human face'

        cv.putText(frame, text, (20,25), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
        cv.putText(frame, str(pred), (500, 25), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0))

        cv.imshow('Camera', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    face_detect()