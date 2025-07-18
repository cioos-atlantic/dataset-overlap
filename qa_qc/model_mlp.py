import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from scipy.stats import skew, kurtosis
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle
import keras
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

eov_range = (-2.0, 40)
eov_col_name = 'temperature'
eov_flag_name = 'qc_flag_temperature'
scaler = MinMaxScaler()

label_mapping = {
    1: "Pass",
    3: "Suspect",
    4: "Error",
    9: "Missing"
}

#Loading Training, Test and Validation Data
ddir_ = os.path.join("CIOOS-Full-Data","chunking")

X1 = pickle.load( open(os.path.join(ddir_, 'Annapolis','Annapolis-X_np_array.pkl'), 'rb'))
y1 = pickle.load( open(os.path.join(ddir_, 'Annapolis','Annapolis-Y_np_array.pkl'), 'rb'))

X2 = pickle.load(open(os.path.join(ddir_, 'Antigonish' ,'Antigonish-X_np_array.pkl'), 'rb'))
y2 = pickle.load(open(os.path.join(ddir_, 'Antigonish','Antigonish-Y_np_array.pkl'), 'rb'))


X3 = pickle.load(open(os.path.join(ddir_,'Cape','Cape-X_np_array.pkl'), 'rb'))
y3 = pickle.load(open(os.path.join(ddir_, 'Cape', 'Cape-Y_np_array.pkl'), 'rb'))

X4 = pickle.load(open(os.path.join(ddir_,'Colchester', 'Colchester-X_np_array.pkl'), 'rb'))
y4 = pickle.load(open(os.path.join(ddir_,'Colchester' ,'Colchester-Y_np_array.pkl'), 'rb'))

X6 = pickle.load(open(os.path.join(ddir_, 'Digby' ,'Digby-X_np_array.pkl'), 'rb'))
y6 = pickle.load(open(os.path.join(ddir_,'Digby' ,'Digby-Y_np_array.pkl'), 'rb'))

X7 = pickle.load(open(os.path.join(ddir_, 'Inverness' ,'Inverness-X_np_array.pkl'), 'rb'))
y7 = pickle.load(open(os.path.join(ddir_, 'Inverness', 'Inverness-Y_np_array.pkl'), 'rb'))

X8 = pickle.load(open(os.path.join(ddir_,'Queens' ,'Queens-X_np_array.pkl'), 'rb'))
y8 = pickle.load(open(os.path.join(ddir_,'Queens','Queens-Y_np_array.pkl'), 'rb'))

X5 = pickle.load(open(os.path.join(ddir_,'Victoria' ,'Victoria-X_np_array.pkl'), 'rb'))
y5 = pickle.load(open(os.path.join(ddir_,'Victoria','Victoria-Y_np_array.pkl'), 'rb'))


X = np.concatenate((X1, X2, X3, X4, X6, X7, X8), axis=0)
y = np.concatenate((y1, y2, y3, y4, y6, y7, y8), axis=0)

valid_X, valid_Y = X5, y5

data_ = np.concatenate((X, y.reshape(-1, 1)), axis=1)

GOOD = 1
mask_g = data_[:, -1] == 1  # PASS samples
good_rows = data_[mask_g]  # filtering PASS Flag rows
mask = data_[:, -1] != 1
other_rows_ = data_[mask]  #filter not PASS Flag rows
unique_unique_rows = np.unique(good_rows, axis=0)   # getting unique rows from PASS Flags
data_ = np.concatenate((unique_unique_rows, other_rows_), axis=0)
np.random.shuffle(data_)
X = data_[:, :-1]
y = data_[:, -1]

X = np.nan_to_num(X, nan=0.0)


print(X.shape)
print(y.shape)



if __name__ == '__main__':

    scaler_x = MinMaxScaler()
    scaler_x = scaler_x.fit(X)
    X = scaler_x.transform(X)

    valid_X = scaler_x.transform(valid_X)

    # Convert labels to one-hot encoding
    y_cat = to_categorical(y)

    print(y_cat.shape)

    # Handle class imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.1, random_state=16)

    # Feedforward neural network (MLP)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(y_cat.shape[1], activation='softmax')
    ])
    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train
    model.fit(X_train, y_train, epochs=30, batch_size=32,
              validation_data=(X_test, y_test),
              class_weight=class_weights_dict, verbose=2)

    # Evaluate
    y_pred_probs = model.predict(X_test, verbose=2)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred, digits=4))


    cm = confusion_matrix(y_true, y_pred)

    print(cm)
    # Optional: get class names if known (else use numbers)
    class_names = [label_mapping[i] for i in sorted(np.unique(y_true))]

    # Plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("testing_cm.png")

    valid_y_pred_probs = model.predict(valid_X, verbose=2)
    valid_y_pred = np.argmax(valid_y_pred_probs, axis=1)
    valid_y_true = valid_Y

    print(classification_report(valid_y_true, valid_y_pred, digits=4))
    cm_valid = confusion_matrix(valid_y_true, valid_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_valid, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Validation Victoria Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("validation_cm.png")