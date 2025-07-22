# 📩 Spam SMS Classifier

This repository implements a machine learning–based spam filter for SMS messages. It classifies messages as either **spam** or **ham** (non-spam).

---

## 🔧 Features

- Data preprocessing: cleaning, normalization (lowercase, removing punctuation, tokenization, stop-word removal, etc.)
- Text vectorization using **CountVectorizer** or **TF‑IDF**
- Model training and evaluation (e.g., **Multinomial Naïve Bayes**, **Logistic Regression**, **SVM**)
- Performance metrics: accuracy, precision, recall, F1-score
- Optional: Streamlit web UI for real-time spam detection

---

## 📁 Repository Structure

```
.
├── data/
│   └── spam.csv                     # Raw SMS dataset (labels + messages)
├── notebooks/
│   └── spam_message_detection.ipynb # EDA, model training & evaluation
├── model/
│   ├── vectorizer.pkl               # Saved CountVectorizer or TF‑IDF
│   └── model.pkl                    # Trained ML model
├── app.py                           # Optional Streamlit or Flask UI
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/alakendu003/Spam_SMS_Classifier.git
   cd Spam_SMS_Classifier
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download and preprocess the dataset**

   Place `spam.csv` in the `data/` folder (if not already included).

4. **Train the model**

   Jupyter notebook:
   ```bash
   jupyter notebook notebooks/spam_message_detection.ipynb
   ```

5. **Run the demo app (optional)**

   ```bash
   streamlit run app.py
   ```

---

## 🧪 Usage Example

```python
from joblib import load

vectorizer = load("model/vectorizer.pkl")
model = load("model/model.pkl")

def classify_sms(text):
    X = vectorizer.transform([text])
    label = model.predict(X)[0]
    return "🛑 Spam" if label == 1 else "✅ Ham"

print(classify_sms("Congratulations! You've won a free ticket!"))
```

---

## 📊 Evaluation Metrics

Typical performance on a standard SMS spam dataset:

| Metric     | Score |
|------------|-------|
| Accuracy   | ~0.98 |
| Precision  | ~0.97 |
| Recall     | ~0.86 |
| F1-Score   | ~0.91 |

---

## 🔄 Customization & Extensions

- Switch vectorizer: Try **TF‑IDF** instead of CountVectorizer
- Try other models: **SVM**, **Random Forest**, **Logistic Regression**
- Tune hyperparameters via grid search or cross-validation
- Add features: phone number detection, uppercase ratio, special characters
- Deploy as API (Flask, FastAPI)

---

## 📚 References

- **UCI SMS‑Spam Collection Dataset**  
- Scikit-learn tutorials on text classification

---

## 👤 Author

`alakendu003`

