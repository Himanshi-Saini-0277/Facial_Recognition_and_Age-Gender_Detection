# Facial Recognition and Age-Gender Detection
This code is designed to perform real-time facial recognition and age-gender detection using OpenCV and a pre-trained Keras model.

#### Requirements:
- OpenCV: Used for accessing the camera feed, detecting faces, and drawing rectangles around them.
- Keras: Utilized for loading a pre-trained deep learning model for facial recognition.
- Caffe Models: Age and gender estimation is performed using pre-trained Caffe models.
- haarcascade_frontalface_default.xml: Haar Cascade classifier for face detection.
- keras_model.h5: Pre-trained Keras model for facial recognition.
- Usage:

#### Face Data Collection:
- The code first prompts the user to enter their name for face data collection.
- It captures images of the user's face and saves them in a directory named after the user.
- It uses the Haar Cascade classifier to detect faces in the video feed.
- The process stops after collecting ten images of the user's face.

#### Real-time Facial Recognition and Age-Gender Detection:

- After collecting the user's face data, it performs real-time facial recognition and age-gender detection.
- It loads the pre-trained Keras model for facial recognition and pre-trained Caffe models for age and gender estimation.
- It continuously captures frames from the camera feed and detects faces using the Haar Cascade classifier.
- For each detected face:
  - It preprocesses the image for input into the facial recognition model.
  - It predicts the identity of the person using the Keras model and draws a rectangle around the face with the predicted identity.
  - It estimates the age and gender of the person using pre-trained Caffe models and displays the results on the screen.
- The process continues until the user presses 'q' to quit the application.

#### Face Data Collection:
###### Working:
1. User Input: The code prompts the user to enter their name for face data collection.
2. Directory Creation: It creates a directory with the user's name to store the collected face images.
3. Capture Loop: The code enters a loop to continuously capture frames from the camera feed.
4. Face Detection: It utilizes the Haar Cascade classifier to detect faces in each frame.
5. Image Saving: For each detected face, it saves the cropped face image in the user's directory.
6. Count Limit: The process stops after collecting ten images of the user's face.

#### Real-time Facial Recognition and Age-Gender Detection:
###### Working:
1. Initialization: The code initializes video capture and sets up parameters for the camera feed.
2. Face Detection: It continuously captures frames from the camera feed and detects faces using the Haar Cascade classifier.
3. Facial Recognition:
  - For each detected face, it preprocesses the image and passes it through a pre-trained Keras model for facial recognition.
  - It draws a rectangle around the detected face and displays the predicted identity on the screen.
4. Age-Gender Detection:
  - It utilizes pre-trained Caffe models for age and gender estimation.
  - It preprocesses the face image and feeds it to the age and gender estimation models.
  - The predicted age and gender labels are displayed alongside the detected face.
5. User Interaction: The process continues until the user presses 'q' to quit the application.
