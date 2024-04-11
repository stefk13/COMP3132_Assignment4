import os
import cv2
import numpy as np
from PIL import Image
import pickle
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


face_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')
profile_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_profileface.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
resized_images_dir = os.path.join(BASE_DIR, "src", "resized-images")
os.makedirs(resized_images_dir, exist_ok=True)

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")

            detected_faces = set()

            for cascade in (face_cascade, profile_cascade):
                faces = cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=4)
                for (x, y, w, h) in faces:
                    detected_faces.add((x, y, x+w, y+h))

            for (x1, y1, x2, y2) in detected_faces:
                roi = image_array[y1:y2, x1:x2]
                aspect_ratio = (x2 - x1) / (y2 - y1)

                if aspect_ratio > 1:
                    new_width = 200
                    new_height = np.int32(new_width / aspect_ratio)
                else:
                    new_height = 200
                    new_width = np.int32(new_height * aspect_ratio)

                roi_resized = cv2.resize(roi, (new_width, new_height))
                padding = (200 - max(new_width, new_height)) // 2

                if new_width > new_height:
                    roi_padded = cv2.copyMakeBorder(roi_resized, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                else:
                    roi_padded = cv2.copyMakeBorder(roi_resized, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

                x_train.append(roi_padded)
                y_labels.append(id_)

                processed_filename = os.path.join(resized_images_dir, f"processed_{label}_{file}")
                processed_image = Image.fromarray(roi_padded)
                processed_image.save(processed_filename)

x_train = np.array(x_train) / 255.0
x_train = x_train.reshape(-1, 200, 200, 1)
y_labels = to_categorical(y_labels)


X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_labels, test_size=0.1, random_state=11)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(label_ids), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, 
                    epochs=10, 
                    validation_data=(X_val, Y_val))

model.save('my_model.keras')
with open("labels_deep.pickle", 'wb') as f:
    pickle.dump(label_ids, f)


# Accuracy Graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig(os.path.join(BASE_DIR, 'training_validation_accuracy.png'))
plt.close()

# Loss Graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig(os.path.join(BASE_DIR, 'training_validation_loss.png'))
plt.close()

# Confusion Matrix
predictions = model.predict(X_val)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(Y_val, axis=1)
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Scale'})
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'))
plt.close()