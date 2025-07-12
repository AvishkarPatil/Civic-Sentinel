# Civic-Sentinel

A computer vision system to detect unusual or damaged objects in civic images, such as broken infrastructure, illegal dumping, or other anomalies.

## Description

Civic-Sentinel uses an autoencoder model to identify anomalies in street-level or public camera imagery. The system is trained on "normal" images of city infrastructure and flags deviations from this baseline. This allows for the automated monitoring and reporting of potential issues, helping municipal teams address problems faster.

---

## Scope of Work

* **Anomaly Detection Model:** Train a deep learning model to distinguish between normal and anomalous civic images.
* **Admin Dashboard:** A simple web interface to upload images for review and see the model's analysis.
* **Alert System:** A basic notification mechanism for when an anomaly is detected.

---

## Tools & Technologies

* **Backend & ML:** Python
* **Deep Learning:** TensorFlow / Keras
* **Web Dashboard:** Streamlit
* **Core Libraries:** OpenCV, NumPy, Matplotlib

---

## Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/AvishkarPatil/Civic-Sentinel.git
    cd Civic-Sentinel
    ```

2.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file will be added later in the project.)*

---

## Usage

1.  **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```

2.  **Open your web browser** and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Upload an image** to see the anomaly detection in action.

---

