import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_dataset(file_path, target_column):
    df = pd.read_csv(file_path)

    # Encode non-numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y
