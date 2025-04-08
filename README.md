# Keyboard-Auth-API

## 📘 Overview

This project presents a real-time user authentication system using **keystroke dynamics** as a biometric modality, combined with **machine learning** techniques. It provides a novel approach to continuous user authentication by comparing real-time keystroke patterns with authenticated user profiles, rather than relying on traditional password-based systems.

By leveraging **FastAPI**, **Python**, and an **Isolation Forest model**, the system offers a secure, software-based solution for identity verification based on behavioral biometrics.

---

## 🔐 Key Features

- Real-time biometric keystroke data capture
- Continuous authentication using behavioral patterns
- Machine learning evaluation for anomaly detection
- FastAPI-based backend with RESTful endpoints
- Lightweight client-side JavaScript integration

---

## 🧠 How It Works

1. **Training Endpoint (`/train-model/`)**:
   - Accepts keystroke data (key, delta time, duration) and a username.
   - Trains or updates the Isolation Forest model for that user.

2. **Evaluation Endpoint (`/evaluate-data/`)**:
   - Accepts new keystroke data and compares it against the user's model.
   - Returns a "normalcy score" indicating whether the data likely belongs to the same user.

---

## 📊 Data Collection & Processing

- Data collected using `keystroke.py` and `keystrokev2.py`.
- Each keypress includes:
  - Key name
  - Time between keypresses
  - Hold duration
- Data stored in CSVs and converted to JSON using `converter.py` and `converterv2.py`.

---

## 🤖 Model Selection

### Initial Experiments:
- Basic neural network (underperformed due to lack of data)
- One-class neural network (low effectiveness)
- Support Vector Machine (failed to generalize)

### Final Model:
- **Isolation Forest** (unsupervised, anomaly-based)
- Chosen for its ability to detect outliers and work with limited labeled data
- Tuned with:
  - `n_estimators = 500`
  - `max_samples = 256`
  - `contamination = 0.073`

---
## 💻 Proof of Concept

- Built a frontend in **HTML/CSS/JavaScript**
- Backend in **Express.js**
- Features:
  - Typing textbox
  - Profile dropdown (Sofia, Jacob, Shiyuan)
  - Toggle for training vs evaluation mode
- Sends data every 256 keypresses to the API
- Console logs normalcy score responses

---

## 🧪 Evaluation Results

- **Correct user**: ~90%+ normalcy score
- **Incorrect user**: ~50–80% normalcy score
- Clear distinction shown in visual graphs using anomaly score thresholds

---

## 📉 Limitations

- Isolation Forest lacks incremental learning
- Limited dataset and user diversity
- External factors (time of day, stress, caffeine) affect typing patterns

---

## 🚀 Future Work

- Integrate neural networks for better feature extraction
- Expand dataset with more diverse participants
- Respect privacy by not logging character data
- Explore modeling external influences on typing patterns

---

## ✅ Conclusion

Keystroke dynamics offer a unique, secure, and continuous method for authenticating users. This project demonstrates the feasibility of using machine learning to identify users based solely on how they type. While our prototype shows promising accuracy, future iterations with larger datasets and advanced models can improve robustness and usability.

---

## 🛠 Tech Stack

- Python
- FastAPI
- Scikit-learn (Isolation Forest)
- JavaScript
- Express.js
- HTML/CSS

---

## 👥 Authors

- Sofia Schnurrenberger
- Jacob 
