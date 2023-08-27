import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('saved_model.h5')

# Define the region of interest (ROI) for digit detection
x, y, w, h = 400, 100, 200, 200

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract the ROI from the frame
    roi = gray[y:y+h, x:x+w]

    # Threshold the ROI to get a binary image
    ret, thresh = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and predict the digit in each contour
    for cnt in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Extract the digit from the binary image
        digit = thresh[y:y+h, x:x+w]

        # Resize the digit to 28x28 pixels
        resized_digit = cv2.resize(digit, (28, 28))

        # Flatten the resized digit to a 1D array
        flattened_digit = resized_digit.reshape(1, 28*28)

        # Scale the flattened digit to the range[2]
        scaled_digit = flattened_digit / 255.0

        # Predict the digit using the trained model
        prediction = model.predict(scaled_digit)

        # Get the predicted digit
        digit_number = np.argmax(prediction)

        # Draw the predicted digit on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, str(digit_number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('frame', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()