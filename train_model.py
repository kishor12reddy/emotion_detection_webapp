import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Use a raw string to avoid unicode escape issues
DIRECTORY = r"C:\Users\DELL\Desktop\ml\FER-2013\train"
CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

data = []
labels = []

# Load images and labels
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    print(f"Loading images from {path}...")  # Debug statement
    if not os.path.exists(path) or len(os.listdir(path)) == 0:
        print(f"No images found in {path}. Please check your dataset.")
        continue
    
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        try:
            # Load and preprocess the image
            image = load_img(img_path, target_size=(128, 128), color_mode="rgb")  # Ensure it's grayscale
            image = img_to_array(image)
            image = image.astype("float32") / 255.0  # Normalize the image
            data.append(image)
            labels.append(CATEGORIES.index(category))  # Use index of the category as label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# Check if data and labels were populated
print(f"Total images loaded: {len(data)}")  # Debug statement
print(f"Total labels loaded: {len(labels)}")  # Debug statement

if len(data) == 0:
    raise Exception("No images were loaded. Please check your dataset.")

# Convert to numpy arrays
data = np.array(data, dtype="float32")
labels = to_categorical(np.array(labels))  # Convert labels to categorical format

# Train-test split
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                         height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))  # Adjust input shape for 128x128 RGB images
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))  # Increased dense layer units for more complex data
model.add(Dropout(0.5))
model.add(Dense(len(CATEGORIES), activation='softmax'))  # Output layer matches number of RAF-DB categories


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Implement early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit the model
history = model.fit(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), epochs=50, callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(testX, testY)
print(f"Test accuracy: {test_acc}")

# Save the trained model
model.save('emotion_model.h5')

