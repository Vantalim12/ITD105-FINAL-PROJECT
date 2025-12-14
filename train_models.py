"""
Fish Species Conservation Status - Model Training and Comparison
Comparing multiple classification algorithms for SDG 14 project
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading dataset...")
df = pd.read_csv('fish_conservation_data.csv')

# Preprocessing
print("\nPreprocessing data...")

# Select features for modeling
feature_columns = ['habitat_type', 'population_trend', 'fishing_pressure', 
                   'average_size_cm', 'geographic_region', 'reproduction_rate',
                   'depth_range_m', 'water_temperature_c', 'population_size_thousands']

X = df[feature_columns].copy()
y = df['conservation_status']

# Encode categorical variables
label_encoders = {}
categorical_columns = ['habitat_type', 'population_trend', 'fishing_pressure', 'geographic_region']

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Split dataset (80% training / 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Normalize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate models
print("\n" + "="*70)
print("TRAINING AND EVALUATING MODELS")
print("="*70)

results = []

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"Training {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"✓ {name} trained successfully!")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)
print(results_df.to_string(index=False))

# Select best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_model_name}")
print(f"Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"{'='*70}")

# Save the best model and preprocessing objects
print("\nSaving model and preprocessing objects...")

model_artifacts = {
    'model': best_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'target_encoder': target_encoder,
    'feature_columns': feature_columns,
    'categorical_columns': categorical_columns
}

with open('best_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

# Save all models and their results for the dashboard
all_models_data = {
    'models': models,
    'results': results_df,
    'X_test': X_test_scaled,
    'y_test': y_test,
    'target_encoder': target_encoder
}

with open('all_models.pkl', 'wb') as f:
    pickle.dump(all_models_data, f)

print("✓ Model artifacts saved successfully!")
print("  - best_model.pkl (for predictions)")
print("  - all_models.pkl (for comparison dashboard)")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

