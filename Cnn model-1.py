#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random


# In[2]:


# Path to your training directorie
train_data_dir = r'C:/Users/DELL 7480/OneDrive/Desktop/PROJECT/images/training data'

# List of class labels (subdirectories) in the training data directory
class_labels = os.listdir(train_data_dir)

# Display random images from the training dataset
plt.figure(figsize=(40,40))
for i, class_label in enumerate(class_labels):
    class_path = os.path.join(train_data_dir, class_label)
    img_names = os.listdir(class_path)
    random_img_name = random.choice(img_names)
    random_img_path = os.path.join(class_path, random_img_name)
    img = load_img(random_img_path, target_size=(420, 420))  # Adjust target_size if needed
    plt.subplot(2, len(class_labels), i + 1)
    plt.imshow(img)
    plt.title(f'Training: {class_label}')
    plt.axis('off')


# In[3]:


train_data_dir = "C:/Users/DELL 7480/OneDrive/Desktop/PROJECT/images/training data"
test_data_dir = "C:/Users/DELL 7480/OneDrive/Desktop/PROJECT/images/testing data"


# In[4]:


batch_size = 32
image_height = 150
image_width = 150
num_classes = 10
epochs = 10


# In[5]:


datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# In[6]:


train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


# In[7]:


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])


# In[8]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[9]:


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
)


# In[22]:


test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)


# In[23]:


# Plot training and testing accuracy
plt.figure(figsize=(10, 6))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


# In[24]:


# Predict test data
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes


# In[25]:


# Accuracy metrics
accuracy = test_accuracy
report = classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys())
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Classification Report:\n", report)


# In[26]:


# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[27]:


#saving the model using it along with opencv 
from keras.models import load_model
# Save the model to a file
model.save("gesture_model.h5") #name of the model


# In[ ]:



