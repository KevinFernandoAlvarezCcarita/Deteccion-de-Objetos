import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
from sklearn import svm
from sklearn.model_selection import train_test_split

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return fd

positive_images_folder = 'positives/'
negative_images_folder = 'negatives/'

positive_images = []
positive_labels = []
for img_name in os.listdir(positive_images_folder):
    img = cv2.imread(os.path.join(positive_images_folder, img_name))
    features = extract_hog_features(img)
    positive_images.append(features)
    positive_labels.append(1)  

negative_images = []
negative_labels = []
for img_name in os.listdir(negative_images_folder):
    img = cv2.imread(os.path.join(negative_images_folder, img_name))
    features = extract_hog_features(img)
    negative_images.append(features)
    negative_labels.append(0)  
X = np.array(positive_images + negative_images)
y = np.array(positive_labels + negative_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

new_image = cv2.imread('new_image.jpg')
new_features = extract_hog_features(new_image)
prediction = clf.predict([new_features])
print(f"Predicción: {'Positivo' if prediction[0] == 1 else 'Negativo'}")
