# 📧 Email Spam Detection using Machine Learning

A machine learning web application that classifies SMS/email messages as **spam** or **ham (non-spam)** using a **Support Vector Classifier (SVC)** with **TF‑IDF vectorization**. The model is trained on a publicly available SMS Spam Collection dataset and deployed as an interactive **Streamlit** app.

🌐 **Live Demo**: [https://email-spam-detection-ml-naved.streamlit.app/](https://email-spam-detection-ml-naved.streamlit.app/)

---

## 📌 Features

- Clean and simple user interface built with Streamlit.
- Real‑time prediction: enter a message and instantly see if it's spam or ham.
- Balanced training data using Random Under‑Sampling to avoid bias.
- High accuracy with SVM and TF‑IDF features.
- Model and vectorizer persisted with `pickle` for fast loading.

---

## 🛠️ Technologies Used

- **Python** (3.8+)
- **Pandas** – data manipulation
- **Scikit‑learn** – TF‑IDF vectorization, SVM model, evaluation, under‑sampling
- **Matplotlib** – data visualization (during exploration)
- **Streamlit** – web app framework
- **Pickle** – model serialization
- **KaggleHub** – dataset download

---

## 📊 Dataset

The dataset is the **SMS Spam Collection** from UCI, downloaded via KaggleHub.  
It contains 5,574 messages labeled as `ham` (legitimate) or `spam`.  
After under‑sampling, the training set becomes perfectly balanced (50% spam, 50% ham).

- Source: [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 🔄 Workflow Diagram

The following diagram illustrates the end‑to‑end machine learning pipeline:

```mermaid
flowchart TD
    A[Load Dataset] --> B[Select Columns & Rename]
    B --> C[Map Labels: ham→0, spam→1]
    C --> D[Balance with RandomUnderSampler]
    D --> E[Train/Test Split 70/30]
    E --> F[TF‑IDF Vectorization]
    F --> G[Train SVM Classifier]
    G --> H[Evaluate on Test Set]
    H --> I[Save Model & Vectorizer]
    I --> J[Build Streamlit App]
    J --> K[Deploy on Streamlit Cloud]
