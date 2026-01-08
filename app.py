import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="B.Tech Placement Predictor - KNN", layout="wide")

st.sidebar.header("Team Details (Group 2)")
st.sidebar.markdown(f"Team Leader: Bibek Nandi")
st.sidebar.markdown("""
Members:
1. Bibek Nandi (14)
2. Anmol Kansal (9)
3. Shubham Kumar (41)
4. Vikash Sharma (54) askvikashsharma@gmail.com
5. Siddhant Singh (42)
""")

st.title("Activity 1: Supervised Learning Review")
st.subheader("Predicting B.Tech CSE Interview Success using KNN")
st.info("Presentation Date: 09/01/2026")

tabs = st.tabs(["1a & 1b: Problem Definition", "1c: Dummy Dataset", "1d: Challenges", "Interactive KNN Model"])

with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        st.header("a. Define the Problem")
        st.write("""
        Objective: To predict whether a B.Tech CSE student will clear a technical interview based on their academic and technical profile.
        
        In the competitive campus placement landscape, we aim to use historical data to identify which students are "Interview Ready" and which require further training.
        """)
    with col2:
        st.header("b. Type of Problem")
        st.success("Type: Supervised Learning (Classification)")
        st.write("""
        - Supervised: Because we use a labeled dataset (past results).
        - Classification: Because the output is a discrete category: Clear or Not Clear.
        """)

with tabs[1]:
    st.header("c. Collected Dataset (30 Samples)")
    
    np.random.seed(42)
    rows = 30
    cgpa = np.round(np.random.uniform(6.0, 9.8, rows), 2)
    dsa = np.random.randint(50, 500, rows)
    projects = np.random.randint(0, 6, rows)
    mock_score = np.random.randint(30, 100, rows)
    
    outcome = []
    for i in range(rows):
        score = (cgpa[i]*10) + (dsa[i]/10) + (projects[i]*5) + (mock_score[i]/2)
        outcome.append(1 if score > 130 else 0)

    df = pd.DataFrame({
        'CGPA': cgpa,
        'DSA_Problems': dsa,
        'Projects': projects,
        'Mock_Score': mock_score,
        'Outcome': outcome
    })
    
    st.dataframe(df, use_container_width=True, height=400)
    st.write(f"Dataset Summary: {df['Outcome'].value_counts().get(1, 0)} Cleared | {df['Outcome'].value_counts().get(0, 0)} Not Cleared")

with tabs[2]:
    st.header("d. Identify Challenges")
    st.error("Key Technical Hurdles in KNN Implementation:")
    st.markdown("""
    1. Feature Scaling: Since 'DSA Problems' (up to 500) and 'CGPA' (0-10) are on different scales, the model would ignore CGPA without StandardScaler.
    2. Choosing K: A small $K$ (like 1) makes the model sensitive to outliers; a large $K$ may include points from the wrong class.
    3. The Curse of Dimensionality: As we add more parameters (Projects, Mock Scores), calculating multi-dimensional distance becomes computationally heavy.
    4. Memory Usage: KNN is a 'Lazy Learner'—it doesn't 'learn' a model but compares new data to all stored samples every time.
    """)

with tabs[3]:
    st.header("Visualizing Model Training & Prediction")
    
    features = ['CGPA', 'DSA_Problems', 'Projects', 'Mock_Score']
    X = df[features] 
    y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    k_val = st.slider("Select value of K", 1, 9, 3, step=2)
    knn = KNeighborsClassifier(n_neighbors=k_val)
    knn.fit(X_scaled, y)
    
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.subheader("Test a New Student")
        in_cgpa = st.number_input("Enter CGPA", 0.0, 10.0, 7.5)
        in_dsa = st.number_input("Enter DSA Problems", 0, 500, 250)
        in_proj = st.slider("Projects", 0, 10, 2)
        in_mock = st.slider("Mock Interview Score", 0, 100, 70)
        
        new_data = np.array([[in_cgpa, in_dsa, in_proj, in_mock]])
        new_data_scaled = scaler.transform(new_data)
        
        if st.button("Predict Result"):
            prediction = knn.predict(new_data_scaled)
            result = "✨ CLEAR" if prediction[0] == 1 else "❌ NOT CLEAR"
            st.metric(label="Interview Prediction", value=result)
            
    with col_b:
        st.subheader("Spatial Distribution (CGPA vs DSA)")
        fig, ax = plt.subplots()
        ax.scatter(df['CGPA'], df['DSA_Problems'], c=df['Outcome'], cmap='coolwarm', edgecolors='k', label="Past Students")
        ax.scatter(in_cgpa, in_dsa, color='yellow', marker='*', s=300, label="New Student", edgecolors='black')
        ax.set_xlabel("CGPA")
        ax.set_ylabel("DSA Problems")
        ax.legend()
        st.pyplot(fig)
        st.caption("Note: Visualization shows only 2 of the 4 parameters used by the model.")
