# Multicamera-video-processing

##  Overview

Developed as a collaborative group project, this system processes multi-angle video recordings of a person and selects the best front-facing frame using pose estimation via MediaPipe. The selected frames are compiled into a final video, followed by evaluation of multiple AI models across Precision, Recall, F1-Score, and Accuracy metrics.

---

##  Key Features

-  Multi-camera processing and synchronization
-  Intelligent best-frame selection using human pose estimation
-  Real-time Euclidean distance calculation and orientation detection
-  Output video compilation with timestamp and camera info
-  Comparative evaluation of 7 AI models across metrics and training percentages
-  Auto-generated graphs and CSV reports for performance analysis

---

##  Project Structure
 project_root/
┣  video1.mp4, video2.mp4, video3.mp4 # Input video files
┣  output_video.mp4 # Combined output video
┣  evaluation_results.csv # Table of metrics
┣  accuracy_results.csv # Model accuracies
┣  main.py # Main processing and evaluation script
┗  README.md # Project documentation

---

## 🧰 Technologies Used

- **Python 3.x**
- **OpenCV**
- **MediaPipe**
- **NumPy**
- **Matplotlib**
- **Pandas**
- **Math**

---

## ⚙️ How It Works

### 📌 Step 1: Pose-Based Frame Selection
- Extracts frames from 3 video feeds.
- Uses MediaPipe Pose to detect body landmarks.
- Applies alignment conditions (shoulders, eyes, nose) and Euclidean distance to determine if the person is **front-facing**.
- Selects the **best frame** and overlays camera info and timestamp.

### 📌 Step 2: Output Video Generation
- All selected frames are compiled into a final video showing the best angle at every timestamp.

### 📌 Step 3: AI Model Evaluation
- Simulates performance for 7 popular AI models on 4 training percentages (60%, 70%, 80%, 90%).
- Metrics: **Precision**, **Recall**, **F1-score**, **Accuracy**
- Saves results as CSVs and displays comparative graphs.

---

## 📊 AI Models Compared

1. Attention-based Fusion Model  
2. 3D-CNN  
3. DECIoT  
4. Mask RCNN + LSTM  
5. V3C  
6. CenterNet  
7. **Proposed FRCNN-DNN** (Highlighted model with highest performance)

---

## 📈 Sample Visuals

- Metric vs Training Percentage plots for each video
- Final accuracy table across models and videos

---

## 🚀 How to Run

1. **Clone the repository:**
    ```bash
    git clone https://github.com/arshiya-anjum-24/multicamera-video-processing.git
    cd multicamera-video-processing
    ```

2. **Install required Python packages:**
    ```bash
    pip install opencv-python mediapipe numpy pandas matplotlib
    ```

3. **Update file paths:**
   - Open `code.py` and update the `video_paths` and `output_path` variables to match your local file locations.

4. **Run the code script:**
    ```bash
    python code.py
    ```
---

## 📄 Output Files

- ✅ `output_video.mp4` – Final compiled video of best views  
- ✅ `evaluation_results.csv` – Evaluation metrics per video  
- ✅ `accuracy_results.csv` – Accuracy table for all models  
- 📊 Auto-generated plots showing model performance

---

## 📌 Contribution

Want to contribute or improve this system? Fork the repo, create a feature branch, and raise a PR!  
Your ideas on extending it to action classification or real-time use cases are welcome.

---

## ✍️ Authors

- **Arshiya Anjum Shaik** – Code Integration, Video Processing  
  [LinkedIn](https://linkedin.com/in/arshiya-anjum-shaik) | [Email](mailto:arshiya.anshaik24@gmail.com)

- **Velaga Mouli** – Evaluation Metrics, Visualization, Model Comparison  
  [LinkedIn](https://linkedin.com/in/mouli-velaga) 

*This project was developed collaboratively as part of an academic group project.*


## 📜 License

This project is under the MIT License.
