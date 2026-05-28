import pandas as pd
import numpy as np
 
np.random.seed(42)
N = 1500
 
education_map = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3,
}
 
all_skills = [
    "Python", "SQL", "Machine Learning", "Data Analysis",
    "Java", "JavaScript", "Cloud (AWS/GCP)", "Power BI",
    "Excel", "Communication", "Leadership", "Project Management"
]
 
departments = ["IT", "Finance", "Marketing", "HR", "Data Science", "Operations", "Sales"]
job_titles = {
    "IT":           ["Software Engineer", "DevOps Engineer", "QA Engineer"],
    "Finance":      ["Financial Analyst", "Accountant", "CFO"],
    "Marketing":    ["Marketing Manager", "Content Writer", "SEO Analyst"],
    "HR":           ["HR Manager", "Recruiter", "Training Specialist"],
    "Data Science": ["Data Scientist", "ML Engineer", "Data Analyst"],
    "Operations":   ["Operations Manager", "Supply Chain Analyst", "Logistics Head"],
    "Sales":        ["Sales Executive", "Account Manager", "Sales Head"],
}
 
records = []
 
for _ in range(N):
    dept = np.random.choice(departments)
    education = np.random.choice(list(education_map.keys()),
                                  p=[0.10, 0.45, 0.35, 0.10])
    edu_score = education_map[education]
 
    max_exp = {0: 20, 1: 25, 2: 20, 3: 15}[edu_score]
    experience = int(np.random.exponential(scale=6, size=1).clip(0, max_exp)[0])
 
    num_skills = np.random.randint(2, 8)
    skills = list(np.random.choice(all_skills, num_skills, replace=False))
    skill_score = num_skills
 
    age = 18 + edu_score * 2 + experience + np.random.randint(0, 4)
    age = min(age, 60)
 
    title = np.random.choice(job_titles[dept])
 
    # Salary logic — realistic rules
    base = 25000
    base += edu_score * 12000
    base += experience * 2500
    base += skill_score * 1500
    dept_bonus = {"Data Science": 15000, "IT": 10000, "Finance": 8000,
                  "Operations": 3000, "Marketing": 2000, "HR": 1000, "Sales": 4000}
    base += dept_bonus.get(dept, 0)
 
    # Random noise
    noise = np.random.normal(0, 8000)
    salary = base + noise
    salary = max(salary, 20000)
 
    label = "High" if salary >= 70000 else "Low"
 
    records.append({
        "Age": age,
        "Education": education,
        "Department": dept,
        "Job_Title": title,
        "Experience_Years": experience,
        "Num_Skills": skill_score,
        "Skills": ", ".join(skills),
        "Salary_USD": round(salary, 2),
        "Salary_Label": label,
    })
 
df = pd.DataFrame(records)
df.to_csv("employee_salary_data.csv", index=False)
print(f"Dataset created: {len(df)} rows")
print(df["Salary_Label"].value_counts())
print(df.head(3))