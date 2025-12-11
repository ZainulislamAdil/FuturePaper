# DeepSecIoT: A LSTM--GRU Based Hybrid Deep Learning Approach with SDN for IoT Attack Profiling
## Description:
This project implements hybrid deep learning models for IoT network traffic classification, focusing on DNS datasets. The models combine LSTM, GRU, CNN, and DNN architectures to improve classification performance for detecting anomalies in IoT traffic.
Three hybrid models are implemented:
1. LSTM + GRU Hybrid Model
2. CNN + GRU Hybrid Model
3. DNN + GRU Hybrid Model
## Dataset Information:
1.  Dataset: dns.csv (IoT DNS traffic dataset)
2.  Features: 34 numerical input features
3.  Target: Binary classification (2 classes)
4.  Preprocessing: Missing values filled with zeros, features scaled using MinMaxScaler
## Code Information:
1. Language: Python 3.11+
2. Frameworks: TensorFlow 2.x / Keras, scikit-learn, matplotlib
## Usage Instructions:

1. Clone or download this repository:
  1.  git clone <repository_url>
   2. cd <repository_folder>

2. Create a virtual environment (recommended):
   python -m venv myenv
   Activate:
   Windows: myenv\Scripts\activate


3. Install required libraries which are mentioned in requirements:
4. Ensure the dataset 'dns.csv' is in the working directory.
5. Run the code:
   python your_script.py
   The script will train LSTM+GRU, CNN+GRU, and DNN+GRU hybrid models sequentially.
## Requirements:

 1. Python 3.11+
 2. import os
 3. import glob
 4. import datetime
 5. import itertools
 6. import numpy 
 7. import pandas
 8. import matplotlib.pyplot
 9. from sklearn.preprocessing import MinMaxScaler
 10. from sklearn.model_selection import train_test_split, StratifiedKFold
 11. from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, matthews_corrcoef
 12. import tensorflow as tf
 13. from tensorflow.keras import Input, Model
 14. from tensorflow.keras.layers import Dense, Flatten, Conv2D, LSTM, GRU, Add, Concatenate
 15. from tensorflow.keras.utils import to_categorical
 16. numpy
 17. pandas
 18. matplotlib
 19. scikit-learn
 20. tensorflow

## Methodology:
1. Data Loading & Preprocessing:
   1. Read CSV using Pandas
   2. Fill missing values
   3. Normalize features using MinMaxScaler
   4. Convert class labels to categorical
2. Model Architectures:
   1. LSTM + GRU: Parallel recurrent branches merged with Add()
   2. CNN + GRU: CNN extracts features; GRU captures temporal dependencies; merged output
   3. DNN + GRU: Dense feedforward network combined with GRU for feature enrichment
3. Training:
   1. Loss: categorical_crossentropy
   2. Optimizer: adam
   3. Batch size: 32, Epochs: 10 (modifiable)
4. Evaluation:
The following evaluation metrics were used to assess the performance of all implemented hybrid deep learning models (LSTM+GRU, CNN+GRU, and DNN+GRU):
1.	Confusion Matrix:
	Used to summarize prediction results and analyze true positives, true negatives, false positives, and false negatives.
2.	Accuracy:
	Measures the overall correctness of the model by calculating the ratio of correctly predicted samples to total samples.
3.	Precision:
	Indicates how many of the predicted positive cases are actually positive, helping to evaluate model reliability in positive classification.
4.	Recall (Sensitivity / True Positive Rate — TPR):
	Measures how many actual positive cases the model correctly identifies.
5.	F1-Score:
	Harmonic mean of precision and recall, providing a balanced metric for imbalanced datasets.
6.	AUC–ROC Curve:
	The Area Under the Receiver Operating Characteristic Curve is used to visualize and compare model classification performance across different threshold settings.
7.	True Negative Rate (TNR / Specificity):
	Measures how well the model identifies actual negative cases.
8.	False Positive Rate (FPR):
	Indicates the proportion of negative samples incorrectly classified as positive.
9.	False Negative Rate (FNR):
	Represents the proportion of positive samples incorrectly classified as negative.

## Citations:
If you use this code or dataset for research, please cite:

* Zain Ul Islam Adil et al., "DeepSecIoT: A LSTM--GRU Based Hybrid Deep Learning Approach with SDN for IoT Attack Profiling", [Journal/Conference Name], 2025
* TON_IoT dataset reference
