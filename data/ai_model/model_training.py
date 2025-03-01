import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load sample data
data = pd.read_csv('../pet_health_data.csv')

# Convert categorical data using one-hot encoding
data = pd.get_dummies(data, columns=['diet', 'stool_appearance'])

# Save the feature column names
feature_columns = data.drop('health_status', axis=1).columns.tolist()

# Split data
X = data.drop('health_status', axis=1)
y = data['health_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and save model WITH feature columns
model_data = {
    'model': RandomForestClassifier(n_estimators=100).fit(X_train, y_train),
    'feature_columns': feature_columns
}

joblib.dump(model_data, 'pet_health_model.pkl')