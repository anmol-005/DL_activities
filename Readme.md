# Predicting B.Tech CSE Interview Success 🎓

This repository hosts **Activity 1: Supervised Learning Review**, a machine learning project focused on predicting whether a B.Tech Computer Science student will clear a technical interview. The project leverages the **K-Nearest Neighbors (KNN)** algorithm and is deployed via a Streamlit web application.

---

## 👥 Team Details (Group 2)

| Role | Name | Roll Number |
| :--- | :--- | :--- |
| **Team Leader** | **Bibek Nandi** | 14 |
| Member | Anmol Kansal | 9 |
| Member | Shubham Kumar | 41 |
| Member | Vikash Sharma | 54 |
| Member | Siddhant Singh | 42 |

---

## 🚀 Project Overview

### 1a. Problem Statement
In the competitive campus placement landscape, identifying "Interview Ready" candidates is a challenge. This project uses historical academic and technical data to predict student outcomes, helping individuals identify areas for improvement before the actual recruitment drive.

### 1b. Type of Problem
- **Category:** Supervised Learning
- **Task:** Classification
- **Logic:** We use a labeled dataset (past student results) to predict a discrete output: **1 (Cleared)** or **0 (Not Cleared)**.

### 1c. Dataset Features
The model analyzes four critical student metrics:
1. **CGPA:** Overall academic performance.
2. **DSA_Problems:** Quantity of Data Structures & Algorithms problems solved.
3. **Projects:** Number of technical projects in the portfolio.
4. **Mock_Score:** Internal mock interview performance.

---

## 📊 Dataset Summary
The model is trained on a collected dataset of **30 samples**:
* **Total Cleared:** 23
* **Total Not Cleared:** 7

---

## ⚠️ Challenges (1d)
* **Data Imbalance:** The distribution is skewed toward "Cleared" results, which can impact model sensitivity.
* **Metric Subjectivity:** Mock interview scores vary based on the interviewer's strictness.
* **Complexity vs. Count:** The number of DSA problems doesn't always reflect the complexity or quality of the student's logic.

---

## 🛠️ Technology Stack
* **Language:** Python
* **ML Library:** Scikit-Learn (K-Nearest Neighbors)
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Cloud

---

## 🔗 Live Demo
You can access the interactive prediction tool here:
**[B.Tech Interview Success Predictor](https://dlactivities-gyomrpnvyuersxuqytageq.streamlit.app/)**

### How to use:
1. Navigate to the link above.
2. Use the sidebar sliders to input student metrics.
3. The KNN algorithm will find the nearest "neighbors" in the dataset to provide an instant prediction.

---
*Created for Activity 1: Supervised Learning Review - 09/01/2026*
