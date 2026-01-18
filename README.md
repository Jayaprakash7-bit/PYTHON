# PYTHON Learning & Projects

A collection of Python scripts, mini-projects, learning exercises, and one complete **SMS Fraud (Spam/Smishing) Detection** machine learning project.

This repository serves as my Python journey — from basic concepts (recursion, classes, simple games) to building a real-world ML classifier deployed as a web app.

## 📁 Repository Structure

- **Learning & Basics**
  - `calculator.py` → Simple command-line calculator
  - `recursion.py` → Examples of recursive functions
  - `rps.py` → Rock-Paper-Scissors game
  - `weightconversion.py` → Weight unit converter
  - `rows and columns.py` → Matrix / grid operations
  - `type of classvariable.py` → Class vs instance variables demo
  - `sample*.py` / `samplechocies.py` / `tempCodeRunnerFile.py` → Various small practice scripts
  - `food.py` → (likely a small food-related script or game)

- **SMS Fraud / Spam Detection Project** (Main Project)
  - `app.py` → Flask web application for SMS fraud detection
  - `enhanced_sms_detector.py` → Core script with model training / prediction
  - `enhanced_sms_detector.pkl` → Trained model file (pickle)
  - `demo_sms_detector.py` → Basic model demo
  - `enhanced_demo.py` → Improved demo version
  - `final_98_accuracy_demo.py` → Final version with high accuracy (~98%)
  - `requirements.txt` → Dependencies (Flask, scikit-learn, nltk, etc.)

- **Documentation**
  - `README.md` → This file
  - `PROJECT_WALKTHROUGH.md` → Detailed project explanation & architecture
  - `QUICK_START_GUIDE.md` → How to run everything quickly
  - `SMS_Fraud_Detection_Project_Documentation.md` → In-depth project report
  - `WEB_DEPLOYMENT_GUIDE.md` → Steps to deploy the web app (Render, Heroku, etc.)
  - `IMAGES_FOR_DOCUMENTATION.md` → Screenshots, architecture diagrams, results

## 🚀 Key Project: SMS Fraud Detection

This is a machine learning-based SMS spam/smishing (SMS phishing) detector built with:

- **Libraries**: scikit-learn, NLTK, TF-IDF vectorization (possibly with some preprocessing enhancements)
- **Model**: Achieved ~98% accuracy on test data (see `final_98_accuracy_demo.py`)
- **Features**: Web interface using Flask (`app.py`)
- **Goal**: Classify incoming SMS as **ham (safe)** or **spam/fraud**

### Quick Start (Local)

1. Clone the repo:
   ```bash
   git clone https://github.com/Jayaprakash7-bit/PYTHON.git
   cd PYTHON
### Tips to improve it further

- Add a badge for Python version or license (e.g., MIT)
- Include a nice GIF/screenshot of the web app (upload images to repo or use imgur)
- If you have a live demo link (after deploying), add it prominently at the top
- Update commit messages to be more descriptive in the future

Let me know if you want to make it shorter, add more sections, or focus only on the SMS fraud project!
