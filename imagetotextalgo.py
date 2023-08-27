import cv2
import os
from google.cloud import vision

# Set the path to your JSON key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'elaborate-tube-388003-b73a6d77fc94.json'

# Instantiate the client
client = vision.ImageAnnotatorClient()

# Provide the IP camera stream URL
ip_camera_url = "https://173.0.220.32:8080/video"

cap = cv2.VideoCapture(ip_camera_url)

# Flag to indicate if the image capture is enabled
capture_enabled = False

def predict(frame):
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
        print("Extracted text:")
        print(extracted_text)
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


