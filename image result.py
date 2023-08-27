import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def process_image(image_path):
    K.clear_session()

    im = Image.open(image_path)
    result = Image.new('RGB', (im.width, im.height), color=(255, 255, 255))
    result.paste(im.convert('RGB'), mask=None)  # Remove transparency mask
    result.save('colors.jpg')

    loaded_model = load_model("saved_model.h5")

    crop_img = get_img_reshape_by_cv2(cv2.imread('colors.jpg'))
    num_pred = []

    for i in range(len(crop_img)):
        if len(crop_img) > 1:
            pred = loaded_model.predict(crop_img[i].reshape(1, 28, 28))
            print(np.argmax(pred))
        else:
            pred = loaded_model.predict(crop_img)
        num_pred.append(np.argmax(pred))

    reversed_list = num_pred[::-1]
    print("This is reversed_list:", reversed_list)

    K.clear_session()
    return num_pred

def get_img_reshape_by_cv2(img_data):
    image = img_data
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    preprocessed_digits = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        roi = cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
        digit = thresh[y:y + h, x:x + w]
        resized_digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
        preprocessed_digits.append(padded_digit)

    inp = np.array(preprocessed_digits)
    return inp

# Example usage
image_path = "images/processed_image.jpg"
prediction = process_image(image_path)
print("This is the model prediction:", prediction)
