# Sentiment-Based Product Recommendation System

An intelligent AI-powered web application that analyzes customer reviews using **Natural Language Processing (NLP)** and **Machine Learning** to predict sentiment and recommend relevant products in real time.

This project integrates sentiment analysis with product category detection to provide context-aware recommendations, making it suitable for real-world **e-commerce analytics platforms**.

---

## Problem Statement

Traditional recommendation systems often suggest products based only on ratings. They fail to understand the **context of user reviews**.  

This system solves that by:

- Analyzing customer sentiment using Machine Learning  
- Detecting product category from user input using NLP  
- Recommending only relevant products  

**Example:**  
Input: "Good air cooler for summer"
Output: Positive sentiment + Only air coolers recommended


---

## Features

- Real-time sentiment prediction (**Positive / Negative**)  
- Confidence score for predictions  
- Intelligent product category detection  
- Smart recommendation engine  
- Interactive web interface using Flask  
- Executive analytics dashboard  
- Model persistence using Joblib  

---

## Technologies Used

| Category       | Tools                           |
|----------------|--------------------------------|
| Language       | Python                         |
| ML             | Scikit-learn                   |
| NLP            | TF-IDF Vectorizer              |
| Model          | Naive Bayes                    |
| Visualization  | Matplotlib, Seaborn            |
| Web Framework  | Flask                          |
| Frontend       | HTML, CSS                      |
| Deployment     | Flask Server                   |

---

## System Architecture

- User Review
- ↓
- Text Preprocessing
- ↓
- TF-IDF Vectorization
- ↓
- Naive Bayes Model
- ↓
- Sentiment Prediction
- ↓
- Keyword-Based Product Filtering
- ↓
- Ranked Recommendations


---

## ⚙ How to Run the Project

1. **Install dependencies**

pip install -r requirements.txt

2. **Run Flask app**

python app.py

3. **Open in browser**

http://127.0.0.1:5000


---

## Machine Learning Pipeline
1.	Text cleaning & normalization
2.	TF-IDF feature extraction
3.	Train-test split
4.	Naive Bayes classification
5.	Evaluation using Accuracy, ROC-AUC, Confusion Matrix
6.	Model saving using Joblib
   
---

## Dashboard & Visual Analytics
The system automatically generates:
•	Sentiment distribution chart
•	ROC curve
•	Confusion matrix
•	Top keywords bar chart
•	Executive performance dashboard
All results are saved inside the Results/ folder.

---

## Business Impact
This system can be used by:
•	E-commerce platforms
•	Customer feedback analytics teams
•	Marketing intelligence systems
It improves recommendation relevance by combining sentiment + product context.

## Author
Rajlaxmi Takmoge



