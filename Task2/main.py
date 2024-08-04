import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
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

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40, 1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.2, epochs=50, batch_size=64)
y_prob = model.predict(x_test)
y_pred = np.argmax(y_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
print("Accuracy:", accuracy_score(y_true, y_pred))
