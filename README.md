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

**Objective:** Extend the analytics system into an agentic AI assessment design assistant that autonomously reasons about quality, retrieves pedagogical best practices, and suggests improvements

**Key Deliverables:**
- **Publicly deployed application** (Link required)
- Agent workflow documentation (States & Nodes)
- Structured assessment improvement report generation
- GitHub Repository & Complete Codebase
- Demo Video (Max 5 mins)
---

