import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap

best_model_location = 'best_model.keras'
model = tf.keras.models.load_model(best_model_location)

class_details = {
    0: "Cardboard",
    1: "Glass",
    2: "Metal",
    3: "Paper",
    4: "Plastic",
    5: "Trash"
}

def classify_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    preprocessed_image = np.expand_dims(image, axis=0)
    predictions = model.predict(preprocessed_image)
    predicted_class_idx = np.argmax(predictions[0])
    class_name = class_details[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    return class_name, confidence

def detect_object():
    ret, frame = video_capture.read()
    if ret:
        class_name, confidence = classify_image(frame)
        print(class_name, confidence)
        cv2.putText(frame, f"{class_name} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Create a new window for displaying the classified image
        image_window = QMainWindow()
        image_window.setWindowTitle("Classified Image")
        image_window.setGeometry(100, 100, 800, 600)

        image_label = QLabel(image_window)
        image_label.setGeometry(10, 10, 640, 480)

        display_image(frame, image_label)

        # Create a reset button to close the window and return to the video
        reset_button = QPushButton('Reset', image_window)
        reset_button.setGeometry(670, 100, 100, 50)
        reset_button.clicked.connect(lambda: image_window.close())

        image_window.show()

def display_image(frame, label=None):
    if label is None:
        label = video_label
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channel = frame.shape
    bytes_per_line = 3 * width
    q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_image)
    label.setPixmap(pixmap)

app = QApplication([])
main_window = QMainWindow()
main_window.setWindowTitle("Object Classification")
main_window.setGeometry(100, 100, 800, 600)

video_capture = cv2.VideoCapture(0)

video_label = QLabel(main_window)
video_label.setGeometry(10, 10, 640, 480)

detect_button = QPushButton('Detect', main_window)
detect_button.setGeometry(670, 100, 100, 50)
detect_button.clicked.connect(detect_object)

main_window.show()

while True:
    ret, frame = video_capture.read()
    if ret:
        display_image(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if the buttons were clicked
    app.processEvents()

video_capture.release()