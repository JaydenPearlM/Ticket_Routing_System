# 📨 IT Ticket Routing System – Machine Learning + FastAPI

This project was developed during my **Break Through Tech (BTT) Sprinternship** as part of a three-week intensive team project internship.  
The system classifies incoming IT support tickets and routes them to the correct department using a **Naive Bayes machine learning model** deployed through a **FastAPI service**.

---

## ✨ Features

- 🔄 **Automated Ticket Classification** – Predicts the correct department for an incoming support ticket.  
- 🧹 **Data Preprocessing & Cleaning** – Includes stopword removal, TF-IDF vectorization, and SMOTE for class balancing.  
- 🤖 **Machine Learning Model** – Naive Bayes classifier trained on preprocessed support ticket data.  
- 📊 **Visualization** – Matplotlib used for data distribution and accuracy visualization.  
- ⚡ **FastAPI Deployment** – Model deployed as an API with endpoints for ticket upload and classification.  
- 🎯 **Performance** – Achieved ~75% accuracy on test dataset.  

---

## 🛠 Tech Stack

**Languages & Environment:** Python, Jupyter Notebook, Google Colab  
**Libraries & Tools:** Pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), Matplotlib  
**Machine Learning Model:** Naive Bayes (MultinomialNB)  
**API Deployment:** FastAPI, Uvicorn  
**Version Control:** Git & GitHub  

---

## 📂 Project Structure

ticket-routing-system/
├── notebooks/
│ └── ticket_routing.ipynb # Jupyter Notebook with data preprocessing & training
├── server/
│ ├── main.py # FastAPI app with prediction endpoint
│ ├── models/ # Saved ML models
│ ├── requirements.txt # Python dependencies
│ └── ...
├── data/
│ └── tickets.csv # Training dataset (not included for confidentiality)
└── README.md

---

## 🚀 Installation & Setup

Clone the repository:

git clone https://github.com/your-username/ticket-routing-system.git
cd ticket-routing-system
Create a virtual environment & install dependencies:


python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r server/requirements.txt
Run the FastAPI server:


cd server
uvicorn main:app --reload
Access the API docs at:
👉 http://127.0.0.1:8000/docs

📊 Model Workflow
Data Cleaning – Remove stopwords, punctuation, and normalize text.
Balancing – Apply SMOTE (Synthetic Minority Oversampling) to handle imbalanced ticket categories.
Vectorization – Convert text into numerical features using TF-IDF.
Training – Fit a Naive Bayes classifier on processed data.
Evaluation – Achieved ~75% accuracy on validation set.


🗺️ Roadmap
 Build Naive Bayes pipeline with TF-IDF
 Handle imbalanced data with SMOTE
 Deploy via FastAPI
 Add advanced ML models (Random Forest, SVM) for comparison

 Integrate logging & monitoring for API usage

 Deploy API on cloud (Render / AWS / GCP)
