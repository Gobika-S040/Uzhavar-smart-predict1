import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

DATA_FILE = "Crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def main():
    data = pd.read_csv(DATA_FILE)
    missing = [c for c in FEATURES + ["label"] if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Model Accuracy on test split: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")
    joblib.dump(model, "crop_model.joblib")
    print("Saved crop_model.joblib")

if __name__ == "__main__":
    main()
