DeepSecIoT: A LSTM–GRU Based Hybrid Deep Learning Approach with SDN for IoT Attack Profiling
========================================================================================

Description
-----------
DeepSecIoT is a hybrid deep learning–based framework designed for IoT network traffic
classification and attack profiling. The project integrates Software Defined Networking (SDN)
concepts with advanced deep learning models to improve the detection of malicious activities
in IoT environments.

The framework combines LSTM, GRU, CNN, and DNN architectures to effectively capture temporal
and spatial features from IoT DNS traffic. Three hybrid deep learning models are implemented
and evaluated for binary attack classification.

Why This Project Is Useful
--------------------------
IoT networks are highly vulnerable to cyberattacks due to resource constraints and large-scale
connectivity. Traditional security mechanisms often fail to detect complex attack patterns.
DeepSecIoT improves intrusion detection by:
- Leveraging hybrid deep learning architectures
- Capturing temporal dependencies in IoT traffic
- Supporting research in IoT security and SDN-based monitoring

Dataset Information
-------------------
Dataset Name: dns.csv
Domain: IoT DNS network traffic
Number of Features: 34 numerical input features
Target Variable: Binary classification (normal / attack)

Preprocessing Steps:
- Missing values are filled with zeros
- Features are normalized using MinMaxScaler
- Class labels are converted to categorical format

Code Information
----------------
Programming Language: Python 3.11+
Frameworks: TensorFlow 2.x / Keras
Libraries: NumPy, Pandas, Matplotlib, scikit-learn

Usage Instructions
------------------
1. Clone the repository
2. Create and activate virtual environment
3. Install dependencies
4. Place dns.csv in project root
5. Run: python your_script.py

Requirements
------------
Python 3.11+
numpy, pandas, matplotlib, scikit-learn, tensorflow

Methodology
-----------
Hybrid LSTM-GRU, CNN-GRU, and DNN-GRU models trained using categorical crossentropy and Adam optimizer.

Evaluation Metrics
------------------
Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix

License
-------
Academic and research use only.

Maintainer
----------
Zain Ul Islam Adil
