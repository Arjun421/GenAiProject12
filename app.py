import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import re

# Page config
st.set_page_config(page_title="Exam Question Analytics", layout="wide")

# Titlegive 
st.title("🎓 ML-Based Exam Question Analytics System")
st.markdown("### Milestone 1: Classical ML & NLP for Question Difficulty Prediction")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset.csv')
    df = df.dropna(subset=['questions', 'model_answer', 'student_answer', 'teacher_marks', 'total_marks'])
    df['difficulty_score'] = df['teacher_marks'] / df['total_marks']
    
    # Create difficulty categories
    def categorize_difficulty(score):
        if score >= 0.8:
            return 'Easy'
        elif score >= 0.5:
            return 'Medium'
        else:
            return 'Hard'
    
    df['difficulty_category'] = df['difficulty_score'].apply(categorize_difficulty)
    return df

# Text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Train models
@st.cache_resource
def train_models(df):
    # Prepare features
    df['combined_text'] = df['questions'] + ' ' + df['model_answer']
    df['processed_text'] = df['combined_text'].apply(preprocess_text)
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['difficulty_category']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=200, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    # Train Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    
    # Calculate metrics
    lr_metrics = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, lr_pred, average='weighted', zero_division=0),
        'report': classification_report(y_test, lr_pred)
    }
    
    dt_metrics = {
        'accuracy': accuracy_score(y_test, dt_pred),
        'precision': precision_score(y_test, dt_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, dt_pred, average='weighted', zero_division=0),
        'report': classification_report(y_test, dt_pred)
    }
    
    return vectorizer, lr_model, dt_model, lr_metrics, dt_metrics, df

# Load data and train
df = load_data()
vectorizer, lr_model, dt_model, lr_metrics, dt_metrics, processed_df = train_models(df)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["📊 Dashboard", "🔮 Predict Difficulty", "📈 Model Performance"])

if page == "📊 Dashboard":
    st.header("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Questions", len(df))
    col2.metric("Average Score", f"{df['difficulty_score'].mean():.2f}")
    col3.metric("Categories", df['difficulty_category'].nunique())
    
    st.subheader("Difficulty Distribution")
    difficulty_counts = df['difficulty_category'].value_counts()
    st.bar_chart(difficulty_counts)
    
    st.subheader("Sample Data")
    st.dataframe(df[['questions', 'difficulty_score', 'difficulty_category']].head(10))

elif page == "🔮 Predict Difficulty":
    st.header("Predict Question Difficulty")
    
    question_input = st.text_area("Enter Question Text:", height=100)
    answer_input = st.text_area("Enter Model Answer:", height=100)
    
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree"])
    
    if st.button("Predict Difficulty"):
        if question_input and answer_input:
            combined = question_input + ' ' + answer_input
            processed = preprocess_text(combined)
            features = vectorizer.transform([processed])
            
            if model_choice == "Logistic Regression":
                prediction = lr_model.predict(features)[0]
                proba = lr_model.predict_proba(features)[0]
            else:
                prediction = dt_model.predict(features)[0]
                proba = dt_model.predict_proba(features)[0]
            
            st.success(f"Predicted Difficulty: **{prediction}**")
            
            st.subheader("Confidence Scores")
            prob_df = pd.DataFrame({
                'Category': lr_model.classes_,
                'Probability': proba
            })
            st.bar_chart(prob_df.set_index('Category'))
        else:
            st.warning("Please enter both question and answer text.")

elif page == "📈 Model Performance":
    st.header("Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logistic Regression")
        st.metric("Accuracy", f"{lr_metrics['accuracy']:.3f}")
        st.metric("Precision", f"{lr_metrics['precision']:.3f}")
        st.metric("Recall", f"{lr_metrics['recall']:.3f}")
        st.text("Classification Report:")
        st.text(lr_metrics['report'])
    
    with col2:
        st.subheader("Decision Tree")
        st.metric("Accuracy", f"{dt_metrics['accuracy']:.3f}")
        st.metric("Precision", f"{dt_metrics['precision']:.3f}")
        st.metric("Recall", f"{dt_metrics['recall']:.3f}")
        st.text("Classification Report:")
        st.text(dt_metrics['report'])

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit, Scikit-Learn & TF-IDF")
