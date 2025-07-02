import pandas as pd
import numpy as np
import pickle
import json 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load the dataset (put diabetic_data.csv in the same folder)
df = pd.read_csv("diabetic_data.csv")

# Convert target to binary
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Clean data
df.replace('?', np.nan, inplace=True)
df.dropna(axis=1, thresh=len(df)*0.5, inplace=True)
df['age'] = LabelEncoder().fit_transform(df['age'].astype(str))

# Feature engineering
df['total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

# Select only the features used in the app
features = ['age', 'time_in_hospital', 'number_inpatient', 'number_emergency', 'number_outpatient', 'total_visits']
X = df[features]
y = df['readmitted']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with class weight
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Save model and feature list
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model_columns.json", "w") as f:
    json.dump(features, f)

print("✅ Model trained and saved with selected features.")
