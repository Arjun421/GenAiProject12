import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
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

# Extract additional features
def extract_features(df):
    features = pd.DataFrame()
    
    # Text length features
    features['question_length'] = df['questions'].str.len()
    features['answer_length'] = df['model_answer'].str.len()
    features['student_answer_length'] = df['student_answer'].str.len()
    
    # Word count features
    features['question_words'] = df['questions'].str.split().str.len()
    features['answer_words'] = df['model_answer'].str.split().str.len()
    features['student_words'] = df['student_answer'].str.split().str.len()
    
    # Complexity indicators
    features['has_explain'] = df['questions'].str.lower().str.contains('explain|describe').astype(int)
    features['has_what'] = df['questions'].str.lower().str.contains('what is|what are').astype(int)
    features['has_define'] = df['questions'].str.lower().str.contains('define|name|list').astype(int)
    features['has_compare'] = df['questions'].str.lower().str.contains('compare|difference|contrast').astype(int)
    
    # Answer similarity (simple ratio)
    features['answer_match_ratio'] = df.apply(
        lambda x: len(set(str(x['student_answer']).lower().split()) & 
                     set(str(x['model_answer']).lower().split())) / 
                 max(len(str(x['model_answer']).split()), 1), axis=1
    )
    
    return features

# Train models
@st.cache_resource
def train_models(df):
    # Prepare text features
    df['combined_text'] = df['questions'] + ' ' + df['model_answer']
    df['processed_text'] = df['combined_text'].apply(preprocess_text)
    
    # TF-IDF vectorization with better parameters
    vectorizer = TfidfVectorizer(
        max_features=150, 
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=2
    )
    X_text = vectorizer.fit_transform(df['processed_text'])
    
    # Extract additional features
    X_extra = extract_features(df)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_extra_scaled = scaler.fit_transform(X_extra)
    
    # Combine TF-IDF and additional features
    from scipy.sparse import hstack
    X_combined = hstack([X_text, X_extra_scaled])
    
    y = df['difficulty_category']
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Logistic Regression with better parameters
    lr_model = LogisticRegression(
        max_iter=500, 
        random_state=42,
        C=1.0,
        class_weight='balanced'
    )
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    # Train Random Forest (better than Decision Tree)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        min_samples_split=5
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    lr_metrics = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, lr_pred, average='weighted', zero_division=0),
        'report': classification_report(y_test, lr_pred)
    }
    
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, rf_pred, average='weighted', zero_division=0),
        'report': classification_report(y_test, rf_pred)
    }
    
    return vectorizer, scaler, lr_model, rf_model, lr_metrics, rf_metrics, df

# Load data and train
df = load_data()
vectorizer, scaler, lr_model, rf_model, lr_metrics, rf_metrics, processed_df = train_models(df)

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
    student_input = st.text_area("Enter Student Answer (optional):", height=100, value="Sample student response")
    
    model_choice = st.selectbox("Select Model", ["Random Forest (Best)", "Logistic Regression"])
    
    if st.button("Predict Difficulty"):
        if question_input and answer_input:
            # Prepare text features
            combined = question_input + ' ' + answer_input
            processed = preprocess_text(combined)
            text_features = vectorizer.transform([processed])
            
            # Prepare additional features
            temp_df = pd.DataFrame({
                'questions': [question_input],
                'model_answer': [answer_input],
                'student_answer': [student_input if student_input else answer_input]
            })
            extra_features = extract_features(temp_df)
            extra_features_scaled = scaler.transform(extra_features)
            
            # Combine features
            from scipy.sparse import hstack
            combined_features = hstack([text_features, extra_features_scaled])
            
            if model_choice == "Random Forest (Best)":
                prediction = rf_model.predict(combined_features)[0]
                proba = rf_model.predict_proba(combined_features)[0]
            else:
                prediction = lr_model.predict(combined_features)[0]
                proba = lr_model.predict_proba(combined_features)[0]
            
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
        st.metric("Accuracy", f"{lr_metrics['accuracy']:.1%}")
        st.metric("Precision", f"{lr_metrics['precision']:.1%}")
        st.metric("Recall", f"{lr_metrics['recall']:.1%}")
        st.text("Classification Report:")
        st.text(lr_metrics['report'])
    
    with col2:
        st.subheader("Random Forest ⭐")
        st.metric("Accuracy", f"{rf_metrics['accuracy']:.1%}")
        st.metric("Precision", f"{rf_metrics['precision']:.1%}")
        st.metric("Recall", f"{rf_metrics['recall']:.1%}")
        st.text("Classification Report:")
        st.text(rf_metrics['report'])
    
    st.info("💡 Random Forest uses 100 decision trees and additional features (question length, keywords, answer similarity) for better accuracy.")

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit, Scikit-Learn & TF-IDF")
