# EduAnalyzer AI: Intelligent Exam Question Analytics System
## From Difficulty Prediction to Agentic Assessment Design Assistant

### Project Overview
This project involves the design and implementation of an **AI-driven exam question analytics system** that predicts question difficulty and evolves into an agentic AI assessment design assistant.

- **Milestone 1:** Classical machine learning and NLP techniques applied to exam questions and student responses to predict difficulty levels and identify learning gaps.
- **Milestone 2:** Extension into an agent-based AI application that autonomously reasons about question quality, retrieves pedagogical best practices (RAG), and generates structured improvement recommendations.

---

### Constraints & Requirements
- **Team Size:** Individual/Group Project
- **API Budget:** Free Tier Only (Open-source models / Free APIs)
- **Framework:** LangGraph (Recommended for M2)
- **Hosting:** Mandatory (Streamlit Cloud, Hugging Face Spaces, or Render)

---

### Technology Stack

| Component | Technology |
| :--- | :--- |
| **ML Models (M1)** | Logistic Regression, Decision Trees (Random Forest, Gradient Boosting), Ensemble |
| **NLP (M1)** | TF-IDF Vectorization (Trigrams), Text Preprocessing |
| **Agent Framework (M2)** | LangGraph, Chroma/FAISS (RAG) |
| **UI Framework** | Streamlit |
| **LLMs (M2)** | Open-source models or Free-tier APIs |

---

### Milestones & Deliverables

#### Milestone 1: ML-Based Exam Question Analytics (Mid-Sem)

**Objective:** Design and implement a machine learning-based exam question analytics system that evaluates difficulty and discrimination using question text and student responses. Focus on classical ML without LLMs.

**Functional Requirements:**
- Accept exam questions and student response data
- Perform text preprocessing and feature extraction
- Predict question difficulty/quality category
- Display analytical insights via user interface

**Technical Requirements (ML):**
- **Preprocessing:** Text cleaning, Encoding, Balanced Dataset
- **Features:** TF-IDF (200 features, trigrams), Response Statistics, Custom Features (22 total)
- **Models:** Logistic Regression, Decision Trees (Random Forest - 200 trees, Gradient Boosting - 150 trees), Ensemble Voting
- **Evaluation:** Accuracy, Precision, Recall, Classification Reports

**Inputs & Outputs:**
- **Input:** Question text, Model answer, Student answer, Teacher marks, Total marks
- **Output:** Difficulty Classification (Easy/Medium/Hard)
- **Metrics:** 92.8% Accuracy achieved

**Key Deliverables:**
- ✅ Problem understanding & Educational context
- ✅ System architecture diagram
- ✅ Working local application with Streamlit UI
- ✅ Model performance evaluation (Accuracy: 92.8%, Precision: 92.8%, Recall: 92.8%)

---

#### Milestone 2: Agentic AI Assessment Design Assistant (End-Sem)

**Objective:** Extend the analytics system into an agentic AI assessment design assistant that autonomously reasons about quality, retrieves pedagogical best practices, and suggests improvements.

**Functional Requirements:**
- Accept assessment design queries
- Analyze difficulty and performance patterns
- Retrieve assessment/pedagogy guidelines (RAG)
- Generate structured improvement recommendations

**Technical Requirements (Agentic):**
- **Agent Framework:** LangGraph with state management
- **RAG System:** Vector database (Chroma/FAISS) with pedagogical best practices
- **Reasoning:** Multi-step analysis of question quality
- **Output:** Structured recommendations with justifications

**Key Deliverables:**
- **Publicly deployed application** (Link required)
- Agent workflow documentation (States & Nodes)
- Structured assessment improvement report generation
- GitHub Repository & Complete Codebase
- Demo Video (Max 5 mins)

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

---

## Features (Milestone 1)

### 📊 Dashboard
- Dataset overview with difficulty distribution
- Average student performance metrics
- Sample question analysis

### 🔮 Predict Difficulty
- Input question text and model answer
- Real-time difficulty prediction
- Confidence score visualization
- Support for 4 models (Ensemble recommended)

### 📈 Model Performance
- Comprehensive metrics for all models
- Classification reports
- Model comparison

---

## Model Performance

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| **🏆 Ensemble** | 92.8% | 92.8% | 92.8% |
| **Gradient Boosting** | 92.8% | 92.8% | 92.8% |
| **Random Forest** | 86.1% | 86.3% | 86.1% |
| **Logistic Regression** | 70.8% | 70.8% | 70.8% |

**Achievement:** Exceeded 80% accuracy target by 12.8%

---

## Technical Implementation

### Feature Engineering (222 Total Features)
- **TF-IDF Features:** 200 features with trigrams
- **Length Features:** Question/answer character and word counts
- **Ratio Features:** Student-to-model answer ratios
- **Keyword Detection:** Complexity indicators (explain, analyze, compare, etc.)
- **Complexity Score:** Weighted sum of difficulty indicators
- **Answer Similarity:** Matching ratio between student and model answers

### Model Architecture
1. **Logistic Regression:** Linear classification with L2 regularization
2. **Random Forest (Decision Tree Ensemble):** 200 decision trees with balanced class weights, parallel voting
3. **Gradient Boosting (Sequential Decision Trees):** 150 decision trees learning sequentially from errors
4. **Ensemble Voting:** Soft voting combining all 3 models with weighted predictions (1:2:2)

### Data Processing
- Balanced dataset using oversampling (equal Easy/Medium/Hard samples)
- Stratified train-test split (80/20)
- Feature scaling with StandardScaler
- Text preprocessing (lowercase, punctuation removal)

---

## Dataset Structure

```csv
questions,model_answer,student_answer,teacher_marks,total_marks
"What is respiration?","Respiration is...","Respiration is...",3.0,5.0
```

**Difficulty Categorization:**
- Easy: Student score ≥ 85%
- Medium: Student score 45-85%
- Hard: Student score < 45%

---

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── Dataset.csv           # Training data
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── packages.txt         # System dependencies
```

---

## Future Work (Milestone 2)

- Implement LangGraph agent workflow
- Add RAG system for pedagogical best practices
- Generate automated question improvement suggestions
- Deploy to Streamlit Cloud
- Create demo video

---

## Contributors

[Your Name/Team Names]

## License

MIT License

---

## Acknowledgments

Built as part of GenAI coursework focusing on ML-based educational analytics and agentic AI systems.
