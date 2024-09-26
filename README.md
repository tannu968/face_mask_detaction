##Project Overview
This project implements a real-time face mask detection system using OpenCV for face detection and Support Vector Machine (SVM) for mask classification. The system detects faces in real-time using a webcam, classifies them as "Mask" or "No Mask," and displays the result on the screen.

Features
Real-time face detection using Haar Cascade Classifier.
Data collection from webcam for faces with and without masks.
Support Vector Machine (SVM) for mask classification.
Dimensionality reduction using Principal Component Analysis (PCA).
Displays mask status ("Mask" or "No Mask") on the screen in real-time.

Project Structure
|-- face_mask_detection/
    |-- with1_mask.npy        # Collected face data with masks
    |-- without_mask.npy       # Collected face data without masks
    |-- face_mask_detection.py # Main script for detection and training
    |-- haarcascade_frontalface_default.xml # Haar Cascade model for face detection
    |-- README.md              # Project documentation
    
##Requirements
Python 3.x
OpenCV
NumPy
Matplotlib
scikit-learn
Installation
Clone the repository:

git clone https://github.com/your-repo/face_mask_detection.git
cd face_mask_detection

pip install opencv-python opencv-contrib-python numpy matplotlib scikit-learn

##How to Run
# Face Detection and Data Collection
To detect faces and collect data, run the following command. The system will detect faces using a webcam, draw rectangles around them, and collect 200 images of faces with and without masks:
python face_mask_detection.py

##Training the Model
The collected images are preprocessed and fed into an SVM classifier for training. Data is labeled as follows:

with_mask.npy (face images with masks)
without_mask.npy (face images without masks)
Training and testing data are split, and PCA is applied for dimensionality reduction before training the SVM model.

##Real-Time Face Mask Detection
After training, the model is used for real-time face mask detection. The webcam detects a face and classifies it as either "Mask" or "No Mask," displaying the result in real-time on the screen.

##Face Detection
haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)

##Data Collection
faces = haar_data.detectMultiScale(img)
face = cv2.resize(face, (50, 50))
data.append(face)

PCA and SVM Classifier
pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
svm = SVC()
svm.fit(x_train, y_train)

PCA is applied to reduce the dimensionality of image data.
SVM is used for classification of masked and unmasked faces.

##Real-Time Detection
pred = svm.predict(face)
cv2.putText(img, n, (x, y), font, 1, (244, 250, 250), 2)

