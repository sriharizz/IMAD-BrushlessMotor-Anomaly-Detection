# 🧠 Industrial Anomaly Detection — Brushless Motor (IMAD-DS)

This project analyzes real industrial sensor data from a brushless motor to detect abnormal machine behavior using **unsupervised machine learning**.  
It mimics a real-world *predictive maintenance system* used in manufacturing to identify potential faults before failure.

---

## 🚀 Overview

- **Dataset:** [IMAD-DS Brushless Motor Dataset](https://www.kaggle.com/datasets/guoqiangwei/industrial-machine-anomaly-detection-datasets)  
  (4,779 `.parquet` recordings — ~4.1 GB total)
- **Goal:** Automatically detect unusual vibration or acoustic patterns that may indicate mechanical faults.
- **Approach:**
  1. Loaded >4,700 time-series sensor recordings.
  2. Extracted signal features — mean, std, RMS, peak-to-peak, kurtosis, skewness, FFT energy ratios.
  3. Trained an **IsolationForest** model to learn "normal" motor patterns.
  4. Assigned each recording an **anomaly score** — higher = more abnormal.
  5. Visualized and ranked the most anomalous signals.

---

## 🧩 Methodology

### 🔹 Data Engineering
Each `.parquet` file contains two columns:
Every recording represents ~83,000 time steps of microphone data from a running brushless motor.

### 🔹 Feature Extraction
For each recording:
- Mean  
- Standard Deviation  
- Root Mean Square (RMS)  
- Peak-to-Peak amplitude  
- Kurtosis  
- Skewness  
- Total FFT Power  
- Low/High Frequency Energy Ratio  

These features compress a long waveform into a single 10-dimensional numeric fingerprint.

### 🔹 Model
- **Algorithm:** IsolationForest (`n_estimators=300`, `contamination=0.02`)
- **Type:** Unsupervised Anomaly Detection
- **Goal:** Identify recordings statistically different from the normal cluster.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Total recordings analyzed | 4,779 |
| Features extracted per file | 10 |
| Anomalies detected (~2%) | ≈95 |
| Output files generated | 3 |

### ✨ Observations
- Normal signals → steady amplitude oscillations near 0  
- Anomalous signals → strong amplitude spikes or energy bursts  
- The model cleanly separates stable and unstable motor behavior

---

## 📈 Sample Visualization

Below are three real sensor recordings detected by the model:
- **Top:** Major anomaly (sharp amplitude burst — likely fault)
- **Middle:** Mild anomaly (early deviation)
- **Bottom:** Normal motor operation (steady waveform)
<img width="1389" height="240" alt="image" src="https://github.com/user-attachments/assets/561179d6-8767-4857-b0c9-4dfd265ec395" />


## 🛠️ Tech Stack

**Languages & Tools:**  
Python · Pandas · NumPy · Scikit-Learn · Matplotlib · Google Colab  

**Machine Learning Techniques:**  
Feature Engineering · Time-Series Analysis · Unsupervised Learning · IsolationForest · Outlier Detection

---

## 📁 Project Structure


---

## 💾 Outputs

| File | Description |
|------|--------------|
| `imad_brushless_features_all_with_scores.parquet` | Extracted features + anomaly scores |
| `imad_top100_anomalies.csv` | Top 100 anomalous files for inspection |
| `imad_brushless_isoforest_all.joblib` | Trained IsolationForest model |
| `Industrial_Anomaly_Detection.ipynb` | End-to-end code notebook |

---

## 💡 Key Insight

The model automatically recognized vibration outliers that would correspond to:
- Bearing damage  
- Rotor imbalance  
- Electrical noise or sudden impact  

This mirrors how real factories monitor and maintain heavy machinery using **AI-driven fault detection systems**.

---

## 🧾 Resume Summary (for quick reference)

> **Industrial Anomaly Detection – Brushless Motor (IMAD-DS)**  
> • Processed 4,700+ acoustic recordings (4.1 GB) from a brushless motor to detect early mechanical faults.  
> • Engineered statistical & spectral features and trained an IsolationForest model for unsupervised anomaly detection.  
> • Identified top 2 % anomalous signals characterized by amplitude spikes.  
> • Visualized waveform comparisons for clear fault interpretation.  
> • *Tech Stack:* Python, Pandas, Scikit-Learn, NumPy, Matplotlib, Google Colab.

---

## 🔗 Repository

📂 **GitHub:** [https://github.com/sriharizz/industrial-anomaly-detection](https://github.com/sriharizz/industrial-anomaly-detection)  
⭐ If you like this project, consider starring the repo!

---

## 🧑‍💻 Author

**Sri Hari**  
Data Science & Machine Learning Enthusiast  
🔗 [GitHub – @sriharizz](https://github.com/sriharizz)
