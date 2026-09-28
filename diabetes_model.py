```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the diabetes dataset
data = pd.read_csv("diabetic_data.csv")

print("Dataset loaded successfully!")
print("Number of rows:", len(data))
print("Number of columns:", len(data.columns))

# Select some useful columns
selected_columns = [
    "race",
    "gender",
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_diagnoses",
    "readmitted"
]

data = data[selected_columns].copy()

# Replace missing values
data = data.replace("?", "Unknown")

# Convert categorical columns into numbers
encoder = LabelEncoder()

categorical_columns = ["race", "gender", "age", "readmitted"]

for column in categorical_columns:
    data[column] = encoder.fit_transform(data[column].astype(str))

# Separate input and target
X = data.drop("readmitted", axis=1)
y = data["readmitted"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
