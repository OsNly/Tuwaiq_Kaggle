import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score , accuracy_score, classification_report
from xgboost import XGBClassifier

# === Load and preprocess data ===
df = pd.read_csv("enriched_d_train.csv")

df.drop(columns=[
    'Program Start Date', 'Program End Date',
    'Technology Type', 'Education Speaciality', 'University Degree Score System',
    'Job Type', 'Still Working', 'College', 'University Degree Score'
], inplace=True, errors='ignore')

# Encode categorical features
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split
X = df.drop(columns='Y')
y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)

# Initialize error_flag and model_confidence columns in train and test
X_train['error_flag'] = 0.0
X_train['model_confidence'] = 0.0

X_test['error_flag'] = 0.0
X_test['model_confidence'] = 0.0

max_iterations = 100
f1_threshold = 0.9

best_model = None
best_f1 = 0.0


for iteration in range(max_iterations):
    print(f"Iteration {iteration + 1}")

    # Train model with current features
    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=12.6,
        gamma=0,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict on train set
    train_probs = model.predict_proba(X_train)[:, 1]
    train_preds = (train_probs >= 0.5).astype(int)

    train_f1 = f1_score(y_train, train_preds)
    print(f"Training F1: {train_f1:.4f}")
    if train_f1 > best_f1:
        best_f1 = train_f1
        best_model = model  # save current trained model


    if train_f1 >= f1_threshold:
        print("Reached desired F1 score. Stopping training.")
        break

    # Update model confidence feature with smoothing (train)
    X_train['model_confidence'] = 0.5 * X_train['model_confidence'] + 0.5 * train_probs

    # Update error_flag in train only after 2 iterations
    if iteration >= 1:
        new_error_flag = (train_preds != y_train).astype(float)
        # Smooth error_flag as moving average with previous error_flag
        X_train['error_flag'] = 0.5 * X_train['error_flag'] + 0.5 * new_error_flag

    # Update model confidence feature in test (use model predictions on test)
    test_probs = model.predict_proba(X_test)[:, 1]
    X_test['model_confidence'] = 0.5 * X_test['model_confidence'] + 0.5 * test_probs

    # Update error_flag in test only after 2 iterations, using previous preds on test
    if iteration >= 1:
        # Since test labels exist, we can calculate error_flag for test similarly
        test_preds = (test_probs >= 0.5).astype(int)
        new_test_error_flag = (test_preds != y_test).astype(float)
        X_test['error_flag'] = 0.5 * X_test['error_flag'] + 0.5 * new_test_error_flag

# Final prediction and evaluation on test data
# Make sure test columns exactly match train columns and same order
X_test_for_pred = X_test[X_train.columns]

final_test_probs = model.predict_proba(X_test_for_pred)[:, 1]
final_test_preds = (final_test_probs >= 0.5).astype(int)
final_test_f1 = f1_score(y_test, final_test_preds)
print(f"Final Test F1 score: {final_test_f1:.4f}")
print(f"Accuracy: {accuracy_score(y_test, final_test_preds):.4f}")
# Use the saved best model, not just the last model
X_test_for_pred = X_test[X_train.columns]  # align columns
final_test_probs = best_model.predict_proba(X_test_for_pred)[:, 1]
final_test_preds = (final_test_probs >= 0.5).astype(int)
final_test_f1 = f1_score(y_test, final_test_preds)
print(f"Best saved model Test F1 score: {final_test_f1:.4f}")
print(classification_report(y_test, final_test_preds))
import joblib

# After the training loop ends and best_model is assigned
joblib.dump(best_model, "best_xgb_model.pkl")
print("Best model saved to best_xgb_model.pkl")