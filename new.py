import cv2
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
print("MobileNetV2 import works!")
import math
from collections import namedtuple
import matplotlib.pyplot as plt
import time

print("Loading MobileNetV2 model...")
model = MobileNetV2(weights='imagenet')
print(" Model loaded.")

Color = namedtuple("Color", ["name", "rgb"])
BASIC_COLORS = [
    Color("black", (0, 0, 0)), Color("white", (255, 255, 255)),
    Color("red", (255, 0, 0)), Color("green", (0, 128, 0)),
    Color("blue", (0, 0, 255)), Color("yellow", (255, 255, 0)),
    Color("orange", (255, 165, 0)), Color("purple", (128, 0, 128)),
    Color("brown", (139, 69, 19)), Color("gray", (128, 128, 128))
]

KNOWN_FRUIT_LABELS = {
    "banana", "orange", "apple", "lemon", "lime", "pomegranate", "fig",
    "grapefruit", "plum", "mango", "guava", "peach", "pear", "apricot",
    "grape", "papaya", "custard_apple", "melon", "watermelon", "berry"
}

def closest_color_name(rgb):
    def distance(c1, c2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    closest = min(BASIC_COLORS, key=lambda c: distance(rgb, c.rgb))
    return closest.name

def load_and_preprocess_image_from_array(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (224, 224))
    processed = preprocess_input(resized.astype(np.float32))
    return image_rgb, np.expand_dims(processed, axis=0)

def predict_fruit(image_tensor):
    preds = model.predict(image_tensor)
    decoded = decode_predictions(preds, top=10)[0]
    return decoded

def dominant_color(image):
    reshaped = image.reshape((-1, 3)).astype(np.float32)
    _, _, palette = cv2.kmeans(
        reshaped, 1, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )
    return tuple(map(int, palette[0]))

if __name__ == "__main__":
    print("📷 Opening external webcam...")

    # Try external webcam first (index 1)
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(" External webcam not found at index 1, trying index 0...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(" No camera found. Please check your webcam connection.")
        exit()

    print("Webcam opened. Press SPACE to capture or wait 3 seconds for auto-capture. Press ESC to quit.")

    captured_frame = None
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to grab frame from camera.")
            break

        cv2.imshow("Camera - SPACE to Capture | ESC to Exit", frame)
        key = cv2.waitKey(1)

        if key % 256 == 27:  # ESC
            print("Exiting without capture.")
            break
        elif key % 256 == 32:  # SPACE
            captured_frame = frame.copy()
            print(" Image captured with SPACEBAR!")
            break
        elif time.time() - start_time > 3:  # Auto capture after 3 sec
            captured_frame = frame.copy()
            print(" Auto-captured image after 3 seconds.")
            break

    cap.release()
    cv2.destroyAllWindows()


    if captured_frame is not None:
        print("Processing captured image...")
        image_rgb, image_tensor = load_and_preprocess_image_from_array(captured_frame)
        predictions = predict_fruit(image_tensor)

        print("\n Top 10 Predictions:")
        for _, label, prob in predictions:
            print(f"{label}: {prob * 100:.2f}%")

        fruits = [(label, prob) for (_, label, prob) in predictions if label.lower() in KNOWN_FRUIT_LABELS]
        print("\n Detected Fruits:")
        if fruits:
            for fruit, prob in fruits:
                print(f"{fruit} ({prob * 100:.2f}%)")
        else:
            print(" No known fruits detected.")

        color = dominant_color(image_rgb)
        color_name = closest_color_name(color)
        print(f"\n Dominant Color: {color_name.capitalize()} (RGB: {color})")

        plt.imshow(image_rgb)
        plt.title("Captured Image")
        plt.axis('off')
        plt.show()
