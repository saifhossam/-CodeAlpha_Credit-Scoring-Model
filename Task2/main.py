import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential ,load_model
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

paths = []
labels = []
for dirname, _, filenames in os.walk(r'E:\Self Study\CodeAlpha\Task 2\TESS Toronto emotional speech set data'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break
print('DataSet loaded')

df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels

sns.countplot(df['label'])
plt.show()

def feature_extraction(filename):
    audio_amp, sr = librosa.load(filename, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=audio_amp, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

features = df['speech'].apply(lambda x: feature_extraction(x))
features = np.array(features.tolist())

splitted_inputs = np.expand_dims(features, -1)
print(splitted_inputs.shape)

enc = OneHotEncoder()
categorical_outputs = enc.fit_transform(df[['label']]).toarray()
print(categorical_outputs.shape)

x_train, x_test, y_train, y_test = train_test_split(splitted_inputs, categorical_outputs, test_size=0.2, random_state=42, shuffle=True)

# model = Sequential([
#     LSTM(256, return_sequences=False, input_shape=(40, 1)),
#     Dropout(0.2),
#     Dense(128, activation='relu'),
#     Dropout(0.2),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(7, activation='softmax')
# ])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(x_train, y_train, validation_split=0.2, epochs=50, batch_size=64)
# y_prob = model.predict(x_test)
# y_pred = np.argmax(y_prob, axis=1)
# y_true = np.argmax(y_test, axis=1)
# print("Accuracy:", accuracy_score(y_true, y_pred))
#
# model.save('emotion_recognition_model.h5')

import tkinter as tk
from tkinter import filedialog, messagebox

# Load the trained model
model = load_model('emotion_recognition_model.h5')

# Define the function to predict emotion
def predict_emotion():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            features = feature_extraction(file_path)
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=-1)
            prediction = model.predict(features)
            predicted_label = np.argmax(prediction, axis=1)
            # Ensure the predicted_label is in the right shape for inverse_transform
            predicted_label_one_hot = np.zeros((1, len(enc.categories_[0])))
            predicted_label_one_hot[0, predicted_label] = 1
            emotion = enc.inverse_transform(predicted_label_one_hot)[0][0]
            result_label.config(text=f"Predicted Emotion: {emotion}")
        else:
            result_label.config(text="No file selected")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
root = tk.Tk()
root.title("Emotion Recognition from Speech")

# Create a button to upload the voice file
upload_button = tk.Button(root, text="Upload Voice File", command=predict_emotion)
upload_button.pack(pady=20)

# Create a label to display the prediction result
result_label = tk.Label(root, text="Predicted Emotion: ", font=("Helvetica", 14))
result_label.pack(pady=20)

# Run the GUI event loop
root.mainloop()