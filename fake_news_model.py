"""
Fake News Detection using Machine Learning
Using Kaggle Fake.csv and True.csv dataset
"""

import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ========================================
# STEP 1: TEXT CLEANING FUNCTION
# ========================================

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


# ========================================
# STEP 2: LOAD & MERGE DATA
# ========================================

print("="*50)
print("FAKE NEWS DETECTION - ML PROJECT")
print("="*50)

print("\nLoading dataset...")

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0   # Fake news
true_df["label"] = 1   # Real news

df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

print("Total samples:", len(df))


# ========================================
# STEP 3: CLEAN TEXT
# ========================================

print("\nCleaning text...")
df["clean_text"] = df["text"].apply(clean_text)


# ========================================
# STEP 4: SPLIT DATA
# ========================================

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ========================================
# STEP 5: FEATURE EXTRACTION (TF-IDF)
# ========================================

print("\nExtracting features...")
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)


# ========================================
# STEP 6: TRAIN MODEL
# ========================================

print("\nTraining model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Training completed!")


# ========================================
# STEP 7: EVALUATION
# ========================================

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))


# ========================================
# STEP 8: SAVE PREDICTIONS
# ========================================

predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

predictions.to_csv("predictions.csv", index=False)
print("\nPredictions saved to predictions.csv")
print("Done!")
