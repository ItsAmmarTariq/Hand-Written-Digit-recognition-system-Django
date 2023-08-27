
import cv2
import librosa
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse
from django.shortcuts import render, redirect
import base64

from tensorflow import keras


import numpy as np
from PIL import Image
import re
from keras.models import load_model
from configmulti import get_img_reshape_by_cv2

from resize_mnist import imageprepare
from keras import backend as K




# Create your views here.

def getHome(request):
    return render(request, 'home.html')


def getDraw(request):
    return render(request, 'canvas.html', {})


def image_solve(request):
    K.clear_session()
    image_data = request.POST['image_data']
    image_data = re.sub("^data:image/png;base64,", "", image_data)
    image_data = base64.b64decode(image_data)

    fh = open("imageToSave.png", "wb")
    fh.write(image_data)
    fh.close()
    print("save success!")
    im = Image.open("imageToSave.png")
    result = Image.new('RGB', (im.width, im.height), color=(255, 255, 255))
    result.paste(im, im)

    result.save('colors.jpg')

    loaded_model = load_model("saved_model.h5")

    crop_img = get_img_reshape_by_cv2(cv2.imread('colors.jpg'))
    num_pred = []

    # pred = loaded_model.predict(crop_img)
    d = {}
    for i in range(len(crop_img)):

        if len(crop_img) > 1:

            pred = loaded_model.predict(crop_img[i].reshape(1, 28, 28))

            print(np.argmax(pred))

        else:

            pred = loaded_model.predict(crop_img)

        num_pred.append(np.argmax(pred))

    num_pred

    for j in range(10):
        d[j] = round(pred[0][j], 3)

    print("this is model pridiciting",num_pred)


    reversed_list = num_pred[::-1]
    print("this is reversed_list",reversed_list)



    kp2 = num_pred
    K.clear_session()
    return render(request, 'display.html',
                  {'an2': kp2})


def voice(request):
    return HttpResponse(" <h2>we are working on voice Dataset</h2>")


# camera
import cv2
import numpy as np
from django.shortcuts import redirect
from tensorflow import keras

def webcam(request):
    model = keras.models.load_model("my_model.h5")
    digit_labels = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9"
    }

    def preprocess_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        image = cv2.resize(image, (28, 28))
        image = image.reshape(1, 28, 28, 1)
        image = image.astype("float32") / 255.0
        return image

    def predict_digit(image):
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction[0])
        digit_label = digit_labels[predicted_class]
        return digit_label

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        cv2.rectangle(frame, (200, 200), (500, 500), (0, 255, 0), 2)
        digit_image = frame[200:500, 200:500]

        if digit_image.any():
            digit_label = predict_digit(digit_image)
            cv2.putText(frame, digit_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        if cv2.waitKey(1) == ord("q"):
            break

        cv2.imshow("Webcam", frame)

    cap.release()
    cv2.destroyAllWindows()

    return redirect("http://127.0.0.1:8000/")


def voice_input(request):
    return HttpResponse("<h2>this is voice input module !</h2>")



def image_input(request):

        import os
        import cv2
        from google.cloud import vision

        # Set the path to your JSON key file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'elaborate-tube-388003-b73a6d77fc94.json'

        # Instantiate the client
        client = vision.ImageAnnotatorClient()

        # Provide the IP camera stream URL
        ip_camera_url = "https://192.168.111.154:8080/video"

        cap = cv2.VideoCapture(ip_camera_url)

        # Flag to indicate if the image capture is enabled
        capture_enabled = False
        result_text = ""

        def predict(frame):
            nonlocal result_text
            # Read the image file
            _, image = cv2.imencode('.jpg', frame)
            content = image.tobytes()

            # Create an image instance
            image = vision.Image(content=content)

            # Enable the OCR feature
            image_context = vision.ImageContext(
                language_hints=["en"]  # Specify the language(s) expected in the image
            )

            # Perform OCR (text detection) on the image
            response = client.text_detection(image=image, image_context=image_context)
            texts = response.text_annotations

            # Process the response
            if texts:
                extracted_text = texts[0].description

                # Filter out non-digit characters
                filtered_text = ''.join(filter(str.isdigit, extracted_text))

                result_text = filtered_text
                print("Extracted text:")
                print(filtered_text)
            else:
                print("No text found in the image.")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow("Video", frame)

            # Check if the 'c' key is pressed to capture an image
            if cv2.waitKey(1) & 0xFF == ord("c"):
                cv2.imwrite("images/captured_image.jpg", frame)
                print("Image captured successfully!")
                capture_enabled = True

            # Check if the 'q' key is pressed to exit the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Check if image capture is enabled and call the predict function
            if capture_enabled:
                predict(frame)
                capture_enabled = False

        cap.release()
        cv2.destroyAllWindows()

        # Pass the result_text to the template
        return render(request, 'ImageResult.html', {'result_text': result_text})


def getVoice(request):
    if request.method == 'POST' and request.FILES.get('document', False):
        K.clear_session()
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        url = fs.url(filename)

        y, sr = librosa.load(url[1:], sr=8000, mono=True)

        mfcc = librosa.feature.mfcc(y, sr=8000, n_mfcc=40)
        if mfcc.shape[1] > 40:
            mfcc = mfcc[:, 0:40]
        else:
            mfcc = np.pad(mfcc, ((0, 0), (0, 40 - mfcc.shape[1])), mode='constant', constant_values=0)

        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = mfcc.reshape((1, 40, 40, 1))

        model = load_model('mnist_sound.h5')

        pred = model.predict(mfcc)

        kp = pred.argmax()
        d = {}
        for i in range(10):
            d[i] = pred[0][i]
            d[i] = round(d[i] * 100, 2)
        K.clear_session()

        return render(request, 'voice_predict.html',
                      {'ans': kp, 'zero': d[0], 'one': d[1], 'two': d[2], 'three': d[3], 'four': d[4], 'five': d[5],
                       'six': d[6], 'seven': d[7], 'eight': d[8], 'nine': d[9]})

    return render(request, 'upload.html')


def loader(request):
    return render(request, 'loader.html')


###  Finger counting module whole code

import cv2
import os
import time
import mediapipe as mp
from django.shortcuts import redirect


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList


def getNumber(ar):
    s = ""
    for i in ar:
        s += str(ar[i])

    if s == "00000":
        return 0
    elif s == "01000":
        return 1
    elif s == "01100":
        return 2
    elif s == "01110":
        return 3
    elif s == "01111":
        return 4
    elif s == "11111":
        return 5
    elif s == "01001":
        return 6
    elif s == "01011":
        return 7


def finger_webcam(request):
    wcam, hcam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wcam)
    cap.set(4, hcam)
    pTime = 0
    detector = handDetector(detectionCon=int(0.75))

    while True:
        success, img = cap.read()
        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=False)

        tipId = [4, 8, 12, 16, 20]
        if len(lmList) != 0:
            fingers = []

            if lmList[tipId[0]][1] > lmList[tipId[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, len(tipId)):
                if lmList[tipId[id]][2] < lmList[tipId[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(getNumber(fingers)), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 20)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)
        cv2.imshow("Webcam", img)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return redirect("http://127.0.0.1:8000/")


