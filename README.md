Fruit Detection with MobileNetV2

This project uses MobileNetV2 (pre-trained on ImageNet) with OpenCV and TensorFlow/Keras to detect and classify fruits in real-time from a webcam feed.  
It also identifies the dominant color of the captured fruit image.


 Features
- Real-time webcam capture (auto-capture after 3 seconds or press SPACE)
- Uses MobileNetV2 to predict the top-10 ImageNet classes.
- Filters predictions to known fruit labels (e.g., apple, banana, orange, mango, etc.).
- Extracts the dominant color of the fruit and maps it to a basic color name.
- Displays the captured image with Matplotlib



