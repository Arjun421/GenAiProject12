import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import re

# Page config
st.set_page_config(page_title="Exam Question Analytics", layout="wide")

# Titlegive 


# Load and preprocess data

def load_data():
    df = pd.read_csv('Dataset.csv')
    df = df.dropna(subset=['questions', 'model_answer', 'student_answer', 'teacher_marks', 'total_marks'])
    
    # More granular difficulty scoring
    df['difficulty_score'] = df['teacher_marks'] / df['total_marks']
    
    # Improved categorization with better thresholds
    def categorize_difficulty(score):
        if score >= 0.85:  # Stricter threshold for Easy
            return 'Easy'
        elif score >= 0.45:  # Adjusted Medium range
            return 'Medium'
        else:
            return 'Hard'
    
    df['difficulty_category'] = df['difficulty_score'].apply(categorize_difficulty)
    
    # Balance dataset by oversampling minority classes
    from sklearn.utils import resample
    
    easy = df[df['difficulty_category'] == 'Easy']
    medium = df[df['difficulty_category'] == 'Medium']
    hard = df[df['difficulty_category'] == 'Hard']
    
    # Find max class size
    max_size = max(len(easy), len(medium), len(hard))
    
    # Oversample to balance
    easy_upsampled = resample(easy, replace=True, n_samples=max_size, random_state=42)
    medium_upsampled = resample(medium, replace=True, n_samples=max_size, random_state=42)
    hard_upsampled = resample(hard, replace=True, n_samples=max_size, random_state=42)
    
    df_balanced = pd.concat([easy_upsampled, medium_upsampled, hard_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced

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
    
    # Length ratios
    features['student_to_model_ratio'] = features['student_words'] / (features['answer_words'] + 1)
    features['answer_to_question_ratio'] = features['answer_words'] / (features['question_words'] + 1)
    
    # Complexity indicators - HARD questions
    features['has_explain'] = df['questions'].str.lower().str.contains('explain|describe|discuss').astype(int)
    features['has_compare'] = df['questions'].str.lower().str.contains('compare|difference|contrast|distinguish').astype(int)
    features['has_analyze'] = df['questions'].str.lower().str.contains('analyze|evaluate|justify|why').astype(int)
    features['has_impact'] = df['questions'].str.lower().str.contains('impact|effect|consequence|result').astype(int)
    
    # EASY question indicators
    features['has_what'] = df['questions'].str.lower().str.contains('what is|what are').astype(int)
    features['has_define'] = df['questions'].str.lower().str.contains('define|name|list|state').astype(int)
    features['has_give'] = df['questions'].str.lower().str.contains('give|name one|name two|name three').astype(int)
    
    # Question complexity score
    features['complexity_score'] = (
        features['has_explain'] * 3 + 
        features['has_compare'] * 3 + 
        features['has_analyze'] * 4 +
        features['has_impact'] * 3 -
        features['has_what'] * 2 - 
        features['has_define'] * 2 -
        features['has_give'] * 2
    )
    
    # Answer similarity (improved)
    features['answer_match_ratio'] = df.apply(
        lambda x: len(set(str(x['student_answer']).lower().split()) & 
                     set(str(x['model_answer']).lower().split())) / 
                 max(len(str(x['model_answer']).split()), 1), axis=1
    )
    
    # Unique word ratio in student answer
    features['student_unique_words'] = df['student_answer'].apply(
        lambda x: len(set(str(x).lower().split())) / max(len(str(x).split()), 1)
    )
    
    # Punctuation count (complex answers have more)
    features['punctuation_count'] = df['student_answer'].str.count('[,;:.]')
    
    return features

# Train models
@st.cache_resource
def train_models(df):
    # Prepare text features
    df['combined_text'] = df['questions'] + ' ' + df['model_answer']
    df['processed_text'] = df['combined_text'].apply(preprocess_text)
    
    # TF-IDF vectorization with better parameters
    vectorizer = TfidfVectorizer(
        max_features=200,  # More features
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams
        min_df=1,
        max_df=0.9,
        sublinear_tf=True  # Use log scaling
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
        max_iter=1000, 
        random_state=42,
        C=0.5,  # Stronger regularization
        class_weight='balanced',
        solver='saga',
        penalty='l2'
    )
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    # Train Random Forest (better than Decision Tree)
    rf_model = RandomForestClassifier(
        n_estimators=200,  # More trees
        max_depth=15,  # Deeper trees
        random_state=42,
        class_weight='balanced',
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Train Gradient Boosting (additional strong model)
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=7,
        random_state=42,
        subsample=0.8
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    
    # Create Ensemble Voting Classifier
    ensemble_model = VotingClassifier(
        estimators=[
            ('lr', lr_model),
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        voting='soft',  # Use probability voting
        weights=[1, 2, 2]  # Give more weight to RF and GB
    )
    ensemble_model.fit(X_train, y_train)
    ensemble_pred = ensemble_model.predict(X_test)
    
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
    
    gb_metrics = {
        'accuracy': accuracy_score(y_test, gb_pred),
        'precision': precision_score(y_test, gb_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, gb_pred, average='weighted', zero_division=0),
        'report': classification_report(y_test, gb_pred)
    }
    
    ensemble_metrics = {
        'accuracy': accuracy_score(y_test, ensemble_pred),
        'precision': precision_score(y_test, ensemble_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, ensemble_pred, average='weighted', zero_division=0),
        'report': classification_report(y_test, ensemble_pred)
    }
    
    return vectorizer, scaler, lr_model, rf_model, gb_model, ensemble_model, lr_metrics, rf_metrics, gb_metrics, ensemble_metrics, df

# Load data and train
df = load_data()
vectorizer, scaler, lr_model, rf_model, gb_model, ensemble_model, lr_metrics, rf_metrics, gb_metrics, ensemble_metrics, processed_df = train_models(df)

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
    
    model_choice = st.selectbox("Select Model", ["🏆 Ensemble (Best - 3 Models)", "Random Forest", "Gradient Boosting", "Logistic Regression"])
    
    if st.button("Predict Difficulty", type="primary"):
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
            
            # Select model
            if model_choice == "🏆 Ensemble (Best - 3 Models)":
                prediction = ensemble_model.predict(combined_features)[0]
                proba = ensemble_model.predict_proba(combined_features)[0]
                model_used = "Ensemble (LR + RF + GB)"
            elif model_choice == "Random Forest":
                prediction = rf_model.predict(combined_features)[0]
                proba = rf_model.predict_proba(combined_features)[0]
                model_used = "Random Forest"
            elif model_choice == "Gradient Boosting":
                prediction = gb_model.predict(combined_features)[0]
                proba = gb_model.predict_proba(combined_features)[0]
                model_used = "Gradient Boosting"
            else:
                prediction = lr_model.predict(combined_features)[0]
                proba = lr_model.predict_proba(combined_features)[0]
                model_used = "Logistic Regression"
            
            # Display prediction with color coding
            if prediction == 'Easy':
                st.success(f"✅ Predicted Difficulty: **{prediction}** (Confidence: {max(proba)*100:.1f}%)")
            elif prediction == 'Medium':
                st.warning(f"⚠️ Predicted Difficulty: **{prediction}** (Confidence: {max(proba)*100:.1f}%)")
            else:
                st.error(f"🔴 Predicted Difficulty: **{prediction}** (Confidence: {max(proba)*100:.1f}%)")
            
            st.caption(f"Model used: {model_used}")
            
            # Show probability distribution
            st.subheader("Confidence Distribution")
            prob_df = pd.DataFrame({
                'Category': ensemble_model.classes_,
                'Probability (%)': proba * 100
            }).sort_values('Probability (%)', ascending=False)
            
            st.bar_chart(prob_df.set_index('Category'))
            
            # Show detailed probabilities
            col1, col2, col3 = st.columns(3)
            for idx, (cat, prob) in enumerate(zip(ensemble_model.classes_, proba)):
                if idx == 0:
                    col1.metric(cat, f"{prob*100:.1f}%")
                elif idx == 1:
                    col2.metric(cat, f"{prob*100:.1f}%")
                else:
                    col3.metric(cat, f"{prob*100:.1f}%")
        else:
            st.warning("Please enter both question and answer text.")

elif page == "📈 Model Performance":
    st.header("Model Performance Metrics")
    
    # Show best model first
    st.subheader("🏆 Ensemble Model (Best)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{ensemble_metrics['accuracy']:.1%}", delta="Best")
    col2.metric("Precision", f"{ensemble_metrics['precision']:.1%}")
    col3.metric("Recall", f"{ensemble_metrics['recall']:.1%}")
    st.text("Classification Report:")
    st.text(ensemble_metrics['report'])
    st.info("💡 Ensemble combines Logistic Regression, Random Forest, and Gradient Boosting with weighted voting for maximum accuracy.")
    
    st.markdown("---")
    
    # Individual models
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Random Forest")
        st.metric("Accuracy", f"{rf_metrics['accuracy']:.1%}")
        st.metric("Precision", f"{rf_metrics['precision']:.1%}")
        st.metric("Recall", f"{rf_metrics['recall']:.1%}")
    
    with col2:
        st.subheader("Gradient Boosting")
        st.metric("Accuracy", f"{gb_metrics['accuracy']:.1%}")
        st.metric("Precision", f"{gb_metrics['precision']:.1%}")
        st.metric("Recall", f"{gb_metrics['recall']:.1%}")
    
    with col3:
        st.subheader("Logistic Regression")
        st.metric("Accuracy", f"{lr_metrics['accuracy']:.1%}")
        st.metric("Precision", f"{lr_metrics['precision']:.1%}")
        st.metric("Recall", f"{lr_metrics['recall']:.1%}")

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit, Scikit-Learn & TF-IDF")
