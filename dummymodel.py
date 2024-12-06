import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Create dummy data
X = np.array([[1], [2], [3], [4], [5]])  # Feature (input)
y = np.array([2, 4, 6, 8, 10])      # Target (output)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a .pkl file
MODEL_PATH = "./models/model.pkl"
with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print(f"Dummy model saved to {MODEL_PATH}")
