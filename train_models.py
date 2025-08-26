import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import os
import json # <-- ADDED: For saving results

# --- PyTorch Model Definition (No changes here) ---
class SimpleNet(nn.Module):
    # ... (rest of the class is the same)
    def __init__(self, input_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# --- MODIFIED: This function now returns the metrics as a dictionary ---
def get_evaluation_results(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    results = {
        "Accuracy": f"{accuracy:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1-Score": f"{f1:.4f}"
    }
    
    print(f"\n--- Evaluation Results for {model_name} ---")
    for metric, value in results.items():
        print(f"{metric}:  {value}")
    print("-" * (28 + len(model_name)))
    
    return results # <-- ADDED: Return the dictionary

def train_and_save_models(data_path="dataset.csv"):
    # --- This section is mostly the same, loading data and splitting it ---
    print("--- Starting Model Training ---")
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1)
    y = df['label']
    print("Dataset loaded successfully.")
    categorical_features = ['protocol_type', 'service', 'flag']
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split for training and testing.")

    # --- ADDED: Dictionary to hold all performance results ---
    all_performance_results = {}

    # Train and Save Random Forest with SMOTE
    print("\nTraining Random Forest model with SMOTE...")
    rf_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    joblib.dump(rf_pipeline, 'random_forest_model.joblib')
    print("Random Forest model saved.")
    
    y_pred_rf = rf_pipeline.predict(X_test)
    # MODIFIED: Get results and store them
    all_performance_results["Random Forest"] = get_evaluation_results("Random Forest (with SMOTE)", y_test, y_pred_rf)

    # Train and Save SVM with SMOTE
    print("\nTraining SVM model with SMOTE...")
    svm_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', SVC(kernel='linear', probability=True, random_state=42))
    ])
    svm_pipeline.fit(X_train, y_train)
    joblib.dump(svm_pipeline, 'svm_model.joblib')
    print("SVM model saved.")
    y_pred_svm = svm_pipeline.predict(X_test)
    # MODIFIED: Get results and store them
    all_performance_results["SVM"] = get_evaluation_results("SVM (with SMOTE)", y_test, y_pred_svm)

    # Train and Save Deep Learning Model with Weighted Loss
    print("\nTraining Deep Learning (PyTorch) model with Weighted Loss...")
    X_train_processed = preprocessor.fit_transform(X_train)
    input_size = X_train_processed.shape[1]
    dl_model = SimpleNet(input_size)
    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(dl_model.parameters(), lr=0.001)
    
    num_epochs = 5
    for epoch in range(num_epochs):
        dl_model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = dl_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"  Epoch {epoch+1}/{num_epochs} complete.")

    joblib.dump(preprocessor, 'preprocessor.joblib')
    torch.save(dl_model.state_dict(), 'deep_learning_model.pth')
    with open("dl_model_input_size.txt", "w") as f:
        f.write(str(input_size))
    print("Deep Learning model and preprocessor saved.")

    dl_model.eval()
    X_test_processed = preprocessor.transform(X_test)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = dl_model(X_test_tensor)
        _, y_pred_dl_tensor = torch.max(outputs, 1)
        y_pred_dl = y_pred_dl_tensor.numpy()
        
    # MODIFIED: Get results and store them
    all_performance_results["Deep Learning"] = get_evaluation_results("Deep Learning (Weighted Loss)", y_test, y_pred_dl)
    
    # --- ADDED: Save all results to a JSON file ---
    with open('model_performance.json', 'w') as f:
        json.dump(all_performance_results, f, indent=4)
    print("\nPerformance metrics saved to `model_performance.json`")

    print("\n--- All models trained, evaluated, and saved successfully! ---")

if __name__ == "__main__":
    train_and_save_models()