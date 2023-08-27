import cv2
import numpy as np


def get_img_reshape_by_cv2(img_data):
    image = img_data
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)  # convert to gray
    ret, thresh = cv2.threshold(grey.copy(), 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours from left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    preprocessed_digits = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        roi = cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y + h, x:x + w]

        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (500, 500))
        # resized_digit = cv2.resize(resized_digit, (500,500))
        resized_digit = cv2.resize(resized_digit, (400, 400))
        resized_digit = cv2.resize(resized_digit, (300, 300))
        resized_digit = cv2.resize(resized_digit, (200, 200))
        resized_digit = cv2.resize(resized_digit, (100, 100))

        resized_digit = cv2.resize(resized_digit, (50, 50))
        resized_digit = cv2.resize(resized_digit, (40, 40))
        resized_digit = cv2.resize(resized_digit, (25, 25))
        resized_digit = cv2.resize(resized_digit, (18, 18))  ##### <------

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        # plt.imshow(padded_digit)

        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digits.append(padded_digit)

    # plt.imshow(padded_digit, cmap="gray")
    # plt.show()

    # print(preprocessed_digits)

    inp = np.array(preprocessed_digits)

    return inp
