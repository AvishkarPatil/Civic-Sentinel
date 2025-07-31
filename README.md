<div align="center">

# 🛣️ Civic Sentinel

**AI-Powered Road Anomaly Detection System**

*Automated detection of potholes and road infrastructure issues using computer vision and machine learning*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red.svg" alt="Framework">
  <img src="https://img.shields.io/badge/ML-Random%20Forest-green.svg" alt="ML Algorithm">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
</p>

</div>

## 📋 Overview

Civic Sentinel is an advanced computer vision system designed to automatically detect road anomalies, specifically potholes, to help municipal authorities maintain better road infrastructure. The system uses machine learning to analyze road images and provide accurate predictions with confidence scores.


#### Demo Video  : https://drive.google.com/file/d/11axw_5vhLn6ePueodgpxo_RSoU81MIPw/view?usp=sharing
## ✨ Features

- 🤖 **AI-Powered Detection** - Random Forest classifier with 95% training accuracy
- 📊 **Interactive Dashboard** - Real-time analytics and visualization
- 📈 **Detection History** - Track and analyze detection patterns
- 📥 **Export Functionality** - Download detection reports as CSV
- 🎯 **High Accuracy** - 92% test accuracy on road anomaly detection

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AvishkarPatil/Civic-Sentinel.git
   cd Civic-Sentinel
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train.py
   ```

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 🏗️ Project Structure

```
Civic-Sentinel/
├── dataset/
│   ├── train/
│   │   ├── Plain/          # Normal road images
│   │   └── Pothole/        # Pothole images
│   └── test/
│       ├── Plain/          # Test normal roads
│       └── Pothole/        # Test potholes
├── anomaly_detector.py     # Core detection model
├── train.py               # Model training script
├── test.py                # Testing script
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
└── README.md
```

## 🔧 Technical Details

- **Algorithm**: Random Forest Classifier
- **Features**: 17 image features (edges, texture, color, contours)
- **Training Data**: Plain roads vs. Pothole classification
- **Performance**: 95% training accuracy, 92% test accuracy
- **Processing**: Real-time image analysis with confidence scores

## 📊 Usage

1. **Upload Image** - Select a road image using the file uploader
2. **AI Analysis** - Click "Analyze Image" for instant detection
3. **View Results** - See prediction, confidence, and detailed analysis
4. **Track History** - Monitor detection patterns over time
5. **Export Data** - Download reports for further analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ by Avishkar Patil
</div>