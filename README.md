
<h1 align="center"> NetGuardAI</h1>
<h3 align="center">Anomaly Detection in Network Traffic using Machine Learning</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Flask-Web%20App-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/ML-Anomaly%20Detection-orange?style=flat-square"/>
</p>

---

##  Project Overview

NetGuardAI is a smart, web-based system for detecting network anomalies—especially Distributed Denial of Service (DDoS) attacks—using multiple machine learning models. It enables dataset upload, synthetic traffic generation, model evaluation, and export of results with rich visualizations.

---

##  Features

-  Upload or generate synthetic `.csv` datasets
-  Train & compare 6 ML models:
  - Isolation Forest (Unsupervised)
  - Random Forest
  - Naive Bayes
  - Logistic Regression
  - SVM
  - Passive-Aggressive Classifier
-  Visual dashboards with:
  - Accuracy comparison bar chart
  - Normal vs DDoS traffic pie chart
  - Time-series plot of traffic behavior
  - Feature importance chart
-  Flask-based responsive web interface

---

##  Tech Stack

| Tech              | Description                     |
|------------------|---------------------------------|
| Python           | Backend logic                   |
| Flask            | Web framework                   |
| Scikit-learn     | ML models & metrics             |
| Pandas, NumPy    | Data handling                   |
| Matplotlib, Seaborn | Visualization                |
| HTML/CSS/JS      | Frontend                        |

---

##  Model Evaluation

| Model                | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Isolation Forest     | 0.66     | 0.00      | 0.00   | 0.00     |
| Random Forest        | 0.74     | 0.58      | 0.54   | 0.56     |
| Naive Bayes          | 0.76     | 0.57      | 0.92   | 0.71     |
| Logistic Regression  | 0.69     | 0.00      | 0.00   | 0.00     |
| SVM                  | 0.69     | 0.00      | 0.00   | 0.00     |

>  **Best Model:** Naive Bayes (high recall & F1)  
>  **Balanced Model:** Random Forest (good accuracy + interpretability)

---

##  Visualizations

-  **Model Accuracy Bar Chart**
-  **Normal vs DDoS Pie Chart**
-  **Time-Series Anomaly Pattern**
-  **Feature Importance Ranking**

---

##  How to Run

```bash
# Clone the repo
git clone https://github.com/snehas2004/Anomaly-Detection.git
cd Anomaly-Detection

# Install dependencies
pip install -r requirements.txt

# Start the app
python app.py
```

Then open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

---

##  Folder Structure

```
NetGuardAI/
├── static/images/       # Chart outputs
├── templates/           # HTML templates
├── models/              # Saved models
├── utils/               # Helper scripts
├── data/                # Uploaded/generated datasets
├── app.py               # Main Flask app
├── requirements.txt
└── README.md
```

---

##  Contributors

- **Sneha Santhosh**
- **Deepasree Pradeep**
- **Devika S**

 Saintgits College of Engineering, Kerala  
 Intel® Unnati AI & ML Program  
 Mentor: Ms. Akshara Sasidharan

---

##  References

- [CICIDS 2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Scikit-learn](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Hands-On ML - O'Reilly](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

---

##  License

This project is part of the Intel® Unnati Industrial Training Program  
For educational and academic use only.

---

##  Project Summary

###  Outcomes
- NetGuardAI successfully detects DDoS attacks in network traffic using a variety of machine learning algorithms.
- Naive Bayes achieved the **highest detection performance** with an F1-score of 0.71 and recall of 0.92.
- Random Forest offered a **balanced accuracy (0.74)** and valuable feature importance insights.
- Visual tools like bar charts, pie charts, and time-series plots improved the interpretability of results.
- The application’s **Flask-based interface** enables smooth interaction, dataset upload, and report exports.

###  Limitations
- Some models (e.g., SVM, Logistic Regression) failed to detect DDoS traffic effectively in the tested dataset.
- The system currently operates in a **batch mode**, not real-time.
- Large `.pkl` files generated during training can exceed GitHub's file size limit.
- No advanced anomaly handling techniques (e.g., deep learning) are integrated yet.

###  Future Scope
- Implement **real-time traffic monitoring** and live anomaly detection pipelines.
- Expand to **deep learning architectures** like LSTM or Autoencoders for better sequence modeling.
- Add **user authentication and logging** for secure and auditable use.
- Integrate with **cloud storage or databases** to handle large-scale datasets and persistent results.
- Include a **self-learning feedback loop** to adapt the model continuously based on user verification.

---
