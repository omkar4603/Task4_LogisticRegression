#  Task 4 – Logistic Regression Classification

##  Objective  
Build a binary classifier using **Logistic Regression** on the **Breast Cancer Wisconsin Dataset** to predict whether a tumor is *malignant* or *benign*.

---

##  Project Structure
Task4_LogisticRegression/
│
├── data/ # Dataset files (if any)
├── results/ # Output plots & metrics
│ ├── confusion.png
│ ├── roc.png
│ ├── sigmoid.png
│ └── metrics.txt
│
├── src/ # Source code files
│ ├── main.py
│ ├── model.py
│ ├── plots.py
│ └── utils.py
│
├── requirements.txt # Dependencies
└── README.md # Project documentation

---

##  Setup Instructions

### 1️ Clone the repository
```bash
git clone <your-repo-link>
cd Task4_LogisticRegression

## Install dependencies
pip install -r requirements.txt

## Run the project
python src/main.py

