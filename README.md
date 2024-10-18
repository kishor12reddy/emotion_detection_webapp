# 🎭 Emotion Detection & Music Recommendation

This project detects human emotions in real-time using a Convolutional Neural Network (CNN) and recommends songs based on the detected emotion. The application uses a webcam feed to capture faces, classifies emotions such as happy, sad, angry, and more, and then provides a list of mood-matching songs.

## 🚀 Features
- **Real-time Emotion Detection**: Captures live video feed using a webcam and detects facial emotions such as:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise
- **Song Recommendations**: After detecting the user's emotion, the program suggests a playlist of songs tailored to that mood.
  
## 📂 Dataset
The emotion detection model is trained on the **RAF-DB (Real-world Affective Faces Database)**, which contains diverse facial expressions annotated for 7 basic emotions:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

You can download the RAF-DB dataset from [here](http://www.whdeng.cn/RAF/model1.html).

## 🧠 Model Architecture
The model uses a Convolutional Neural Network (CNN) with the following layers:
- Three convolutional layers followed by MaxPooling
- A fully connected Dense layer with 256 neurons and a dropout of 0.5 to avoid overfitting
- A final output layer with softmax activation to classify the emotion

### Libraries Used
- **TensorFlow** for deep learning and building the CNN model
- **OpenCV** for face detection and real-time video feed
- **NumPy** for data manipulation
- **Keras** for high-level neural network API (integrated with TensorFlow)
- **Scikit-learn** for splitting the dataset into training and testing sets

## 🖥️ Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kishor12reddy/emotion-detection-webapp
   cd emotion-detection-webapp
   ```

2. **Install the required libraries**:
   Make sure you have Python 3.x installed and install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare the dataset**:
   - Download the **RAF-DB dataset** from the official website: [RAF-DB Dataset](http://www.whdeng.cn/RAF/model1.html)
   - Extract the dataset into your local directory and organize it like this:
     ```
     RAF-DB/
       ├── train/
           ├── angry/
           ├── disgust/
           ├── fear/
           ├── happy/
           ├── neutral/
           ├── sad/
           ├── surprise/
     ```

4. **Train the Model**:
   Run the `train_model.py` script to train the CNN model:
   ```bash
   python train_model.py
   ```
   The model will be saved as `emotion_model.h5` once training is complete.

5. **Run the Emotion Detector**:
   To start the real-time emotion detection with song recommendations, run the `emotion_detector.py` script:
   ```bash
   python emotion_detector.py
   ```

## 🎶 Song Recommendation
The program recommends songs based on the detected emotion, using predefined playlists in the following categories:. You can also add songs according to your wish
- **Happy**: "Happy - Pharrell Williams", "Uptown Funk - Mark Ronson feat. Bruno Mars", etc.
- **Sad**: "Someone Like You - Adele", "Fix You - Coldplay", etc.
- **Angry**: "Killing in the Name - Rage Against the Machine", etc.
- **Surprise**: "Uptown Funk - Mark Ronson", "Shake It Off - Taylor Swift", etc.

## 📝 Example Usage
1. Run the script to activate the webcam.
2. The model detects the user's face and classifies their emotion in real-time.
3. The detected emotion is displayed on the video feed.
4. A playlist is recommended based on the detected emotion.

## 🎯 Future Improvements
- Expand the song recommendation playlists with more diverse options.
- Improve the emotion detection accuracy by using a more complex model.
- Add support for detecting multiple faces and emotions simultaneously.
## 🧑‍💻 Author
https://github.com/kishor12reddy

Feel free to contribute to this project by submitting issues or pull requests!
