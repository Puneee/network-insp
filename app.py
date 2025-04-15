#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Author: Puneetha Suryadevara

#ABOUT: This is a Machine Learning System that polices Network Traffic. It uses traditional and deep learning methods, to compare the results for better probability of accuracy.

#In Detail this project uses 3 models:

#Random Forest

#Support Vector Machine (SVM)

#Autoencoder (unsupervised deep learning)

#Using these models it outputs:

#For each model (RF, SVM, Autoencoder), you now have:

#Accuracy: how many it got right

#Precision & Recall: how good it is at catching attacks

#Confusion Matrix: how many true/false positives/negatives

#Top features (Random Forest only)

#Anomaly thresholding (Autoencoder only)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#Libraries being used:

import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 1. Download the NSL-KDD dataset using KaggleHub

dataset_name = "cyberdeeplearning/nsl-kdd-dataset"
path = kagglehub.dataset_download(dataset_name)

# Print the dataset path and file contents to see the number of columns
print("Dataset path:", path)
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))

# Assign train and test file paths
train_data_path = os.path.join(path, "NSL_KDD_Train.csv")
test_data_path = os.path.join(path, "NSL_KDD_Test.csv")

# 2. Define column names (42 total: 41 features + 1 label)

column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label"
]


# 3. Load the datasets

train_df = pd.read_csv(train_data_path, names=column_names)
test_df = pd.read_csv(test_data_path, names=column_names)

# Check for missing values
print("\nMissing values:")
print(train_df.isnull().sum())


# 4. Encode categorical columns

categorical_cols = ['protocol_type', 'service', 'flag']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    encoders[col] = le

# Convert label column to binary (normal=0, attack=1)
train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Split into X and y
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']


# 5. Random Forest Classifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=["Normal", "Attack"]))
print(confusion_matrix(y_test, y_pred_rf))

# Plot top 10 feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title("Top 10 Important Features")
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), X_train.columns[indices[:10]], rotation=45)
plt.tight_layout()
plt.show()


# 6. Support Vector Machine

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

print("\nSVM Results")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, target_names=["Normal", "Attack"]))
print(confusion_matrix(y_test, y_pred_svm))


# 7. Autoencoder Anomaly Detection

X_train_ae = X_train[y_train == 0]  # Train only on normal traffic
scaler_ae = StandardScaler()
X_train_ae_scaled = scaler_ae.fit_transform(X_train_ae)
X_test_scaled = scaler_ae.transform(X_test)

input_dim = X_train_ae_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(
    X_train_ae_scaled, X_train_ae_scaled,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# Predict and calculate reconstruction error
reconstructions = autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 95)
print("Autoencoder Threshold:", threshold)

y_pred_ae = (mse > threshold).astype(int)

print("\nAutoencoder Results")
print("Accuracy:", accuracy_score(y_test, y_pred_ae))
print(classification_report(y_test, y_pred_ae, target_names=["Normal", "Attack"]))
print(confusion_matrix(y_test, y_pred_ae))
