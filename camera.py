# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
#
# # Load the pre-trained MNIST model
# model = keras.models.load_model("my _model.h5")
#
# # Create a dictionary to map the predicted class indices to their corresponding digit labels
# digit_labels = {
#     0: "0",
#     1: "1",
#     2: "2",
#     3: "3",
#     4: "4",
#     5: "5",
#     6: "6",
#     7: "7",
#     8: "8",
#     9: "9"
# }
#
# # Define a function to preprocess the input image for the model
# def preprocess_image(image):
#     # Convert the image to grayscale
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Resize the image to match the input size of the model (28x28 pixels)
#     image = cv2.resize(image, (28, 28))
#     # Reshape the image to be a 4D tensor (batch_size, height, width, channels)
#     image = image.reshape(1, 28, 28, 1)
#     # Convert the pixel values to floats between 0 and 1
#     image = image.astype("float32") / 255.0
#     return image
#
# # Define a function to predict the digit in an image
# def predict_digit(image):
#     # Preprocess the input image
#     preprocessed_image = preprocess_image(image)
#     # Use the model to make a prediction
#     prediction = model.predict(preprocessed_image)
#     # Get the index of the predicted class
#     predicted_class = np.argmax(prediction[0])
#     # Get the corresponding digit label from the dictionary
#     digit_label = digit_labels[predicted_class]
#     return digit_label
#
# # Open the webcam and start capturing frames
# cap = cv2.VideoCapture(0)
#
# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()
#     # Flip the frame horizontally
#     frame = cv2.flip(frame, 1)
#     # Draw a rectangle on the frame to show the area where the digit should be written
#     cv2.rectangle(frame, (200, 200), (500, 500), (0, 255, 0), 2)
#     # Crop the area of the frame where the digit should be written
#     digit_image = frame[200:500, 200:500]
#     # Check if the digit is present inside the rectangle
#     if digit_image.any():
#         # Predict the digit in the image
#         digit_label = predict_digit(digit_image)
#         # Display the predicted digit label on the frame
#         cv2.putText(frame, digit_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
#     # Check if the user has pressed the 'q' key to quit the program
#     if cv2.waitKey(1) == ord("q"):
#         break
#     # Display the frame with the predicted digit label
#     cv2.imshow("Webcam", frame)
#
# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()
#
#
