# MLCybersecurity


###Network Intrusion Detection with Machine Learning###
This project demonstrates how to build and compare different machine learning models for a Network Intrusion Detection System (NIDS). The goal is to classify network traffic as either 'normal' or an 'anomaly' (attack).

The models are trained on fake generated NSL-KDD dataset, a benchmark dataset for intrusion detection.

Tech Stack:

Python: Core programming language.

scikit-learn: For data preprocessing and training the Random Forest and SVM models.

PyTorch: For building and training the deep learning model (a simple neural network).

Pandas: For data manipulation.

Streamlit: To create an interactive dashboard that simulates a real-time monitoring system.

Project Structure

network-intrusion-detection/
├── train_models.py       # Script to preprocess data, train, and save the models.
├── app.py                # The Streamlit dashboard application.
├── requirements.txt
└── README.md

How It Works

Data Preparation: The generated NSL-KDD dataset is loaded and preprocessed. This involves one-hot encoding categorical features, scaling numerical features, and creating a binary label ('normal' vs. 'anomaly').

Model Training (train_models.py):

A Random Forest Classifier is trained.

A Support Vector Machine (SVM) is trained on a smaller subset of the data (as it's computationally intensive).

A Deep Learning model (a simple feed-forward neural network) is built and trained using PyTorch.

All trained models and the data scaler are saved to disk.

Real-time Simulation (app.py):

A Streamlit dashboard loads the saved models and the scaler.

It provides an interface where a user can input sample network connection features.

The dashboard preprocesses the input in the same way as the training data and feeds it to each model.

Each model's prediction (Normal or Anomaly) is displayed, allowing for a direct comparison of their outputs.

##Results##


--- Evaluation Results for Random Forest ---
Accuracy:  1.0000
Precision: 1.0000
Recall:    1.0000
F1-Score:  1.0000
-----------------------------------------


--- Evaluation Results for SVM ---
Accuracy:  1.0000
Precision: 1.0000
Recall:    1.0000
F1-Score:  1.0000
----------------------------------------


--- Evaluation Results for Deep Learning (PyTorch) ---
Accuracy:  0.7500
Precision: 0.6667
Recall:    1.0000
F1-Score:  0.8000
---------------------------------------------------

Comments:
Based on the evaluation, the data balancing strategy was highly effective, completely resolving the models' initial failure to learn. The Random Forest and SVM models demonstrated flawless performance, achieving perfect 1.0 scores across all metrics, indicating they could perfectly classify every instance in this specific test set. While these results are excellent, such perfection on a small dataset can suggest a risk of overfitting. In contrast, the deep learning model delivered a more realistic and still very strong performance, achieving a perfect recall of 1.0—meaning it successfully identified every single attack—at the cost of a lower precision, which shows it produced some false alarms. Overall, while the classical models were the top performers, the deep learning model's results better reflect a typical trade-off in real-world cybersecurity applications.
