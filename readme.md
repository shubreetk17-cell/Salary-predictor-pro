💼 Employee Salary Prediction

A Machine Learning project that predicts whether an employee falls in the High or Low salary category using a Decision Tree Classifier.

📁 Project Structure

salary_prediction/

│

├── generate_dataset.py     ← Creates 1,500 realistic employee records

├── train_model.py          ← Trains Decision Tree & saves model

├── app.py                  ← Streamlit web application

├── requirements.txt        ← Python dependencies

│

├── employee_salary_data.csv  (auto-generated)

├── salary_model.pkl          (auto-generated)

├── dept_encoder.pkl          (auto-generated)

└── model_meta.json           (auto-generated)

🚀 How to Run

Step 1 — Install Dependencies

pip install -r requirements.txt

Step 2 — Generate Dataset

python generate_dataset.py

Step 3 — Train the Model

python train_model.py

Step 4 — Launch the App

streamlit run app.py

Open your browser at: http://localhost:8501

🎯 Features Used
Feature	Description
Age	Employee's age (18–60)
Education	High School / Bachelor's / Master's / PhD
Experience	Years of work experience (0–25)
Num_Skills	Number of technical/soft skills
Department	IT, Finance, HR, Data Science, etc.


📊 Model Performance
Metric	Score
Test Accuracy	~79%
5-Fold CV Accuracy	~81% ± 2%
Algorithm	Decision Tree (Gini)
Max Depth	8


🌳 How the Decision Tree Works

Root → Age

  ├── Young (< threshold)

  │     └── Education

  │           ├── Low Education → LOW SALARY

  │           └── High Education → Check Skills → HIGH / LOW

  └── Experienced (> threshold)

        └── Department

              ├── Data Science / IT → HIGH SALARY

              └── HR / Marketing → Check Experience → HIGH / LOW

🛠️ Tech Stack
Python 3.10+
Scikit-learn — Decision Tree
Pandas / NumPy — Data Processing
Streamlit — Web Interface
Plotly — Interactive Charts


Threshold: Salary ≥ $70,000/year → HIGH | Below → LOW
 