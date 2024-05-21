import numpy as np
import cv2
from keras.models import load_model

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

age1 = "age_deploy.prototxt"
age2 = "age_net.caffemodel"
gen1 = "gender_deploy.prototxt"
gen2 = "gender_net.caffemodel"

age_net = cv2.dnn.readNet(age2, age1)
gender_net = cv2.dnn.readNet(gen2, gen1)

model = load_model('keras_model.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LABELS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LABELS = ['Male', 'Female']
CONFIDENCE_THRESHOLD = 0.75

font = cv2.FONT_HERSHEY_COMPLEX

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    image = np.reshape(image, (1, 224, 224, 3))
    return image

def get_className(classNo):
    if classNo == 0:
        return 'Johnny_Depp'
    elif classNo == 1:
        return 'Arnold'
    elif classNo == 2:
        return 'Jim_Carrey'
    elif classNo == 3:
        return 'Emma_Watson'
    elif classNo == 4:
        return 'Queen_Elizabeth'
    else:
        return 'Unknown'

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, imgOriginal = cap.read()
    if not success:
        break

    faces = faceDetect.detectMultiScale(imgOriginal, 1.3, 5)

    for (x, y, w, h) in faces:
        try:
            crop_img = imgOriginal[y:y+h, x:x+w]

            img = preprocess_image(crop_img)

            prediction = model.predict(img)
            classIndex = np.argmax(prediction)
            probabilityValue = np.amax(prediction)

            if probabilityValue > CONFIDENCE_THRESHOLD:
                className = get_className(classIndex)
            else:
                className = "Unknown"

            cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(imgOriginal, f'{className} {round(probabilityValue * 100, 2)}%', (x, y - 10), font, 0.75, (255, 255, 255), 2)

            blob = cv2.dnn.blobFromImage(crop_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            age_net.setInput(blob)
            agePreds = age_net.forward()
            age = AGE_LABELS[agePreds[0].argmax()]

            gender_net.setInput(blob)
            genderPreds = gender_net.forward()
            gender = GENDER_LABELS[genderPreds[0].argmax()]

            cv2.putText(imgOriginal, f'{gender}, {age}', (x, y + h + 20), font, 0.75, (255, 0, 0), 2)

        except Exception as e:
            print(f"Error processing face: {e}")

    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
