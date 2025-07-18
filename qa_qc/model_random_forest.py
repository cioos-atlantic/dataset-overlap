import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

from qa_qc.ai_utils import QartodFlags

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
if __name__ == '__main__':


    ds_p_dir_ =  "D:/CIOOS-Full-Data/chunking/"

    # Loading Training, Test and Validation Data
    Xs = [
        'Annapolis/Annapolis-X_np_array.pkl',
          'Antigonish/Antigonish-X_np_array.pkl',
          # 'Inverness-X_np_array.pkl',
          # 'Queens-X_np_array.pkl',
          # 'Colchester/Colchester-X_np_array.pkl',
          # 'Digby/Digby-X_np_array.pkl',
        'Cape/Cape-X_np_array.pkl',
        'Victoria/Victoria-X_np_array.pkl',
    ]
    Ys = [
        'Annapolis/Annapolis-Y_np_array.pkl',
          'Antigonish/Antigonish-Y_np_array.pkl',
          # 'Inverness-Y_np_array.pkl',
          # 'Queens-Y_np_array.pkl',
        'Cape/Cape-Y_np_array.pkl',
          'Victoria/Victoria-Y_np_array.pkl',
          # 'Colchester/Colchester-Y_np_array.pkl',
          # 'Digby/Digby-Y_np_array.pkl',

    ]

    lst_X = []
    lst_Y = []
    for ei in range(len(Xs)):
        Xpath = os.path.join(ds_p_dir_ , Xs[ei])
        yPath = os.path.join(ds_p_dir_, Ys[ei])
        X1 = pickle.load(open(Xpath, 'rb'))
        y1 = pickle.load(open(yPath, 'rb'))
        lst_X.append(X1)
        lst_Y.append(y1)


    X = np.concatenate(tuple(lst_X[0:-1]), axis=0)
    y = np.concatenate(tuple(lst_Y[0:-1]), axis=0)

    valid_X, valid_Y = lst_X[-1], lst_Y[-1]

    data_ = np.concatenate((X, y.reshape(-1, 1)), axis=1)


    mask_g = data_[:, -1] == QartodFlags.GOOD  # PASS samples
    good_rows = data_[mask_g]  # filtering PASS Flag rows
    mask_o = data_[:, -1] != QartodFlags.GOOD
    other_rows_ = data_[mask_o]  # filter not PASS Flag rows

    num_rows_to_select = other_rows_.shape[0] * 10

    unique_unique_rows = np.unique(good_rows, axis=0)  # getting unique rows from PASS Flags
    print(f"GOOD: {unique_unique_rows.shape}")
    random_row_indices = np.random.choice(unique_unique_rows.shape[0], size=num_rows_to_select, replace=False)

    # Extract the random rows
    unique_unique_rows = unique_unique_rows[random_row_indices, :]
    print(f"GOOD After resample: {unique_unique_rows.shape}")

    data_ = np.concatenate((unique_unique_rows, other_rows_), axis=0)
    np.random.shuffle(data_)
    X = data_[:, :-1]
    y = data_[:, -1]

    X = np.nan_to_num(X, nan=0.0)

    print(X.shape)
    print(y.shape)

    # scaler_x = MinMaxScaler()
    # scaler_x = scaler_x.fit(X)
    # X = scaler_x.transform(X)

    # valid_X = scaler_x.transform(valid_X)

    X = np.round(X, 2)
    valid_X = np.round(valid_X, 2)

    print(X.shape)
    print(y.shape)

    # Step 1: Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    n_estimatr = 40
    # Step 2: Train Random Forest
    rf = RandomForestClassifier(n_estimators=n_estimatr, random_state=7, class_weight='balanced')
    rf.fit(X_train, y_train)

    # Step 3: Predictions
    y_pred = rf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    # Step 4: Evaluation
    print("Validation Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Optional: Feature importances
    import matplotlib.pyplot as plt

    importances = rf.feature_importances_
    plt.figure(figsize=(10, 5))
    plt.bar(range(X.shape[1]), importances)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title(f"RF Feature Importances: Estimator {n_estimatr}")
    plt.show()

    class_names = [label_mapping[i] for i in sorted(np.unique(y_test))]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    valid_y_pred = rf.predict(valid_X)
    cm = confusion_matrix(valid_Y, valid_y_pred)
    print(cm)
    print(classification_report(valid_Y, valid_y_pred, digits=4))
    print("Accuracy:", accuracy_score(valid_Y, valid_y_pred))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Validation Victoria Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()
