import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_dataset

datasets = {
    "heart": ("datasets/Heart.csv", "target"),
    "diabetes": ("datasets/Diabetes.csv", "Outcome"),
    "kidney": ("datasets/Kidney.csv", "class"),
    "liver": ("datasets/Liver.csv", "Diagnosis"),
    "cancer": ("datasets/Cancer.csv", "diagnosis"),
}

for disease, (file, target) in datasets.items():
    print(f"Training model for {disease}...")
    X, y = preprocess_dataset(file, target)
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, f"models/{disease}_model.pkl")

print("✅ All models trained and saved!")
