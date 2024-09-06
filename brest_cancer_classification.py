import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    
    def __init__(self, num_inputs, learning_rate=0.01):
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = learning_rate
    
    def linear(self, inputs):
        Z = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return Z
    
    def Heaviside_step_fn(self, z):
        return 1 if z >= 0 else 0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def tanh(self, z):
        return np.tanh(z)
        
    def predict(self, inputs, activation='Heaviside_step_fn'):
        Z = self.linear(inputs)
        if isinstance(Z, np.ndarray):
            if activation == 'sigmoid':
                return [self.sigmoid(z) for z in Z]
            elif activation == 'Heaviside_step_fn':
                return [self.Heaviside_step_fn(z) for z in Z]
            elif activation == 'tanh':
                return [self.tanh(z) for z in Z]
        else:
            if activation == 'sigmoid':
                return self.sigmoid(Z)
            elif activation == 'Heaviside_step_fn':
                return self.Heaviside_step_fn(Z)
            elif activation == 'tanh':
                return self.tanh(Z)
    
    def loss(self, prediction, target):
        return target - prediction
    
    def train(self, inputs, target, activation):
        prediction = self.predict(inputs, activation)
        error = self.loss(prediction, target)
        self.weights[1:] += self.learning_rate * error * inputs
        self.weights[0] += self.learning_rate * error
        
    def fit(self, X, y, num_epochs, activation):
        for epoch in tqdm(range(num_epochs), desc=f"Training with {activation}"):
            for inputs, target in zip(X, y):
                self.train(inputs, target, activation)

# Load the data from CSV file
data = pd.read_csv('breast-cancer.csv')

# Drop rows with missing values and unnecessary columns
data = data.dropna()
data = data.drop('id', axis=1)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
X = data.drop('diagnosis', axis=1).values
y = data['diagnosis'].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define different learning rates and activation functions to test
learning_rates = [0.001, 0.01, 0.015, 0.016, 0.1, 1.0]
activations = ['Heaviside_step_fn', 'sigmoid', 'tanh']
accuracies = {activation: [] for activation in activations}

for activation in activations:
    for lr in learning_rates:
        # Initialize and train the perceptron with the current learning rate and activation function
        perceptron = Perceptron(num_inputs=X_train.shape[1], learning_rate=lr)
        perceptron.fit(X_train, y_train, num_epochs=1000, activation=activation)
        
        # Make predictions on the test set
        y_pred = perceptron.predict(X_test, activation)
        
        # Convert predictions to binary values if using sigmoid or tanh
        if activation in ['sigmoid', 'tanh']:
            y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
        
        # Calculate accuracy and store the result
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[activation].append(accuracy)
        print(f'Activation Function: {activation}, Learning Rate: {lr}, Accuracy: {accuracy * 100:.2f}%')

# Plot the accuracy for different learning rates and activation functions
plt.figure(figsize=(12, 8))
for activation in activations:
    plt.plot(learning_rates, accuracies[activation], marker='o', label=activation)

plt.xscale('log')  # Use logarithmic scale for better visualization
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Perceptron Accuracy vs Learning Rate for Different Activation Functions')
plt.legend()
plt.grid(True)
plt.show()
