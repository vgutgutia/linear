import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student_performance_dataset.csv")
data = df[['Study_Hours_per_Week', 'Attendance_Rate', 'Final_Exam_Score']].dropna()

X = data[['Study_Hours_per_Week', 'Attendance_Rate']].values
y = data['Final_Exam_Score'].values.reshape(-1, 1)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

theta = np.random.randn(X_b.shape[1], 1)
alpha = 0.0001
epochs = 500
m = len(y)

def compute_cost(X, y, theta):
    return (1/(2*m)) * np.sum((X.dot(theta) - y)**2)

initial_theta = theta.copy()
initial_cost = compute_cost(X_b, y, theta)

for _ in range(epochs):
    gradients = (1/m) * X_b.T.dot(X_b.dot(theta) - y)
    theta -= alpha * gradients

final_cost = compute_cost(X_b, y, theta)

print("Starting Parameters:", initial_theta.flatten())
print("Learning Rate:", alpha)
print("Epochs:", epochs)
print("Starting Cost:", initial_cost)
print("Final Parameters:", theta.flatten())
print("Final Cost:", final_cost)
