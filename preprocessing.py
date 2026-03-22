import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset/creditcard.csv")

# Separate features and target
X = df.drop('Class', axis=1)   # everything except Class
y = df['Class']                # only Class column

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print shapes to verify
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
#“I split the dataset into training and testing sets using an 80-20 ratio to evaluate model performance and avoid overfitting