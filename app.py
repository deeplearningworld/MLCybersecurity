import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import os
import json 

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

# --- Load Pre-trained Models (No changes here) ---
@st.cache_resource
def load_models():
    # ... (rest of the function is the same)
    try:
        rf_model = joblib.load('random_forest_model.joblib')
        svm_model = joblib.load('svm_model.joblib')
        preprocessor = joblib.load('preprocessor.joblib')
        with open("dl_model_input_size.txt", "r") as f:
            input_size = int(f.read())
        dl_model = SimpleNet(input_size)
        dl_model.load_state_dict(torch.load('deep_learning_model.pth'))
        dl_model.eval()
        return rf_model, svm_model, preprocessor, dl_model
    except FileNotFoundError:
        return None, None, None, None

# --- Streamlit App UI ---
st.title("Network Intrusion Detection System ")

# Load the models
rf_model, svm_model, preprocessor, dl_model = load_models()

if rf_model is None:
    st.error(
        "**Error: Model files not found!**\n\n"
        "Please run the `train_models.py` script first to train and save the models:\n\n"
        "```bash\n"
        "python train_models.py\n"
        "```"
    )
else:
    # --- ADDED: New section to display model performance ---
    st.header("Model Performance on Test Data")
    
    try:
        with open('model_performance.json', 'r') as f:
            performance_data = json.load(f)
        
        # Create columns for a nice layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Random Forest")
            rf_perf = performance_data.get("Random Forest", {})
            for metric, value in rf_perf.items():
                st.metric(label=metric, value=value)

        with col2:
            st.subheader("SVM")
            svm_perf = performance_data.get("SVM", {})
            for metric, value in svm_perf.items():
                st.metric(label=metric, value=value)
        
        with col3:
            st.subheader("Deep Learning")
            dl_perf = performance_data.get("Deep Learning", {})
            for metric, value in dl_perf.items():
                st.metric(label=metric, value=value)
    
    except FileNotFoundError:
        st.warning("`model_performance.json` not found. Run `train_models.py` to see performance metrics.")
    
    st.markdown("---") # Add a separator

    # --- User Input Sidebar (No changes here) ---
    st.sidebar.header("Connection Features for Live Prediction")
    protocol_type = st.sidebar.selectbox("Protocol Type", ['tcp', 'udp', 'icmp'])
    service = st.sidebar.selectbox("Service", ['http', 'ftp_data', 'smtp', 'private', 'other', 'ecr_i', 'telnet', 'domain_u'])
    flag = st.sidebar.selectbox("Flag", ['SF', 'S0', 'REJ', 'RSTO', 'RSTR', 'other'])
    src_bytes = st.sidebar.number_input("Source Bytes", min_value=0, value=300)
    dst_bytes = st.sidebar.number_input("Destination Bytes", min_value=0, value=1500)
    count = st.sidebar.number_input("Count", min_value=0, value=5)
    dst_host_srv_count = st.sidebar.number_input("Dst Host Srv Count", min_value=0, value=25)
    dst_host_same_srv_rate = st.sidebar.slider("Dst Host Same Srv Rate", 0.0, 1.0, 0.1)

    if st.sidebar.button("Predict"):
        # Create a DataFrame from the user's input
        input_data = pd.DataFrame({
            # This large dictionary remains the same
            'duration': [0], 'protocol_type': [protocol_type], 'service': [service], 'flag': [flag], 'src_bytes': [src_bytes], 'dst_bytes': [dst_bytes],'land': [0], 'wrong_fragment': [0], 'urgent': [0], 'hot': [0], 'num_failed_logins': [0], 'logged_in': [1], 'num_compromised': [0],'root_shell': [0], 'su_attempted': [0], 'num_root': [0], 'num_file_creations': [0], 'num_shells': [0], 'num_access_files': [0],'num_outbound_cmds': [0], 'is_host_login': [0], 'is_guest_login': [0],'count': [count], 'srv_count': [count], 'serror_rate': [0.0], 'srv_serror_rate': [0.0], 'rerror_rate': [0.0], 'srv_rerror_rate': [0.0],'same_srv_rate': [1.0], 'diff_srv_rate': [0.0], 'srv_diff_host_rate': [0.0],'dst_host_count': [dst_host_srv_count], 'dst_host_srv_count': [dst_host_srv_count], 'dst_host_same_srv_rate': [dst_host_same_srv_rate], 'dst_host_diff_srv_rate': [0.0],'dst_host_same_src_port_rate': [0.0], 'dst_host_srv_diff_host_rate': [0.0],'dst_host_serror_rate': [0.0], 'dst_host_srv_serror_rate': [0.0],'dst_host_rerror_rate': [0.0], 'dst_host_srv_rerror_rate': [0.0]
        })

        # --- Make Predictions (No changes here) ---
        st.header("Live Prediction Results")
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            rf_pred = rf_model.predict(input_data)[0]
            rf_result = "Anomaly" if rf_pred == 1 else " Normal"
            st.metric("Random Forest", rf_result)
        
        with pred_col2:
            svm_pred = svm_model.predict(input_data)[0]
            svm_result = " Anomaly" if svm_pred == 1 else "Normal"
            st.metric("SVM", svm_result)

        with pred_col3:
            processed_input = preprocessor.transform(input_data)
            dl_input_tensor = torch.tensor(processed_input, dtype=torch.float32)
            with torch.no_grad():
                dl_output = dl_model(dl_input_tensor)
                _, dl_pred_idx = torch.max(dl_output.data, 1)
            dl_result = "Anomaly" if dl_pred_idx.item() == 1 else " Normal"
            st.metric("Deep Learning", dl_result)

    # Display the training dataset in an expander
    with st.expander("View the Training Data (`dataset.csv`)"):
        try:
            df = pd.read_csv("dataset.csv")
            st.dataframe(df)
        except FileNotFoundError:
            st.write("`dataset.csv` not found.")