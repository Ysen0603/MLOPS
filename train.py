import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
import json
from datetime import datetime

# Load dataset
df = pd.read_csv('ks-projects-201801.csv')

# Feature Engineering
df['day_duration'] = (pd.to_datetime(df['deadline']) - pd.to_datetime(df['launched'])).dt.days
numerical_features = ['usd_goal_real', 'day_duration','usd_pledged_real','pledged']
categorical_features = ['main_category', 'country']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Définir le nom du modèle
model_name = "RandomForestClassifier"
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Preparing data for training
X = df[numerical_features + categorical_features]
y = (df['state'] == 'successful').astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predict probabilities
y_pred = pipeline.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Structure des métriques améliorée
metrics = {
    "model_info": {
        "name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": {
            "numerical": numerical_features,
            "categorical": categorical_features
        }
    },
    "performance_metrics": {
        "accuracy": accuracy,
        "detailed_metrics": {
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"]
        },
        "class_metrics": report
    }
}

# Sauvegarder avec un nom de fichier qui inclut le modèle
filename = f"metrics_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(filename, "w") as outfile:
    json.dump(metrics, outfile, indent=4)

print(f"Model: {model_name}")
print(f"Accuracy: {accuracy}")
print(f"Results saved in: {filename}")
