#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:	Kantuxue (https://github.com/chaoswork)
Date: Wed Jul 31 07:37:57 2024
Brief: 计算前向和反向传播的 FLOPs
"""


from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class ThreeLayerNN:
    """
    使用纯 numpy 实现的简单的三层网络。
    """
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.W1 = np.random.randn(input_size, hidden1_size) * 0.01
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
        self.W3 = np.random.randn(hidden2_size, output_size) * 0.01

        self.b1 = np.zeros((1, hidden1_size))
        self.b2 = np.zeros((1, hidden2_size))
        self.b3 = np.zeros((1, output_size))

        self.first_forward = True
        self.first_backward = True
        self.forward_flops = 0
        self.backward_flops = 0

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.X = X
        self.batch_size = X.shape[0]

        if self.first_forward:
            print(f"B={self.batch_size},INPUT={self.X.shape[1]},H1={self.W1.shape[1]},H2={self.W2.shape[1]}")
            print("\nForward Propagation:")
            print(f"{'Layer':<15}{'Operation':<20}{'Computation':<30}{'FLOP forward':<20}")
            print("-" * 85)

        self.A1 = np.dot(X, self.W1) + self.b1
        if self.first_forward:
            flops = 2 * X.shape[1] * self.W1.shape[1] * self.batch_size
            print(f"{'Input':<15}{'A1=W1@X':<20}{'2*input*hidden1*batch':<30}{flops:<20}")
            self.forward_flops += flops

        self.A1R = self.relu(self.A1)
        if self.first_forward:
            flops = self.A1.size
            print(f"{'ReLU':<15}{'A1R=ReLU(A1)':<20}{'hidden1*batch':<30}{flops:<20}")
            self.forward_flops += flops

        self.A2 = np.dot(self.A1R, self.W2) + self.b2
        if self.first_forward:
            flops = 2 * self.W1.shape[1] * self.W2.shape[1] * self.batch_size
            print(f"{'Hidden1':<15}{'A2=W2@A1R':<20}{'2*hidden1*hidden2*batch':<30}{flops:<20}")
            self.forward_flops += flops

        self.A2R = self.relu(self.A2)
        if self.first_forward:
            flops = self.A2.size
            print(f"{'ReLU':<15}{'A2R=ReLU(A2)':<20}{'hidden2*batch':<30}{flops:<20}")
            self.forward_flops += flops

        self.A3 = np.dot(self.A2R, self.W3) + self.b3
        if self.first_forward:
            flops = 2 * self.W2.shape[1] * self.W3.shape[1] * self.batch_size
            print(f"{'Hidden2':<15}{'A3=W3@A2R':<20}{'2*hidden2*output*batch':<30}{flops:<20}")
            self.forward_flops += flops

        self.A3R = self.relu(self.A3)
        if self.first_forward:
            flops = self.A3.size
            print(f"{'RELU':<15}{'A3R=RELU(A3)':<20}{'output*batch':<30}{flops:<20}")
            self.forward_flops += flops

        self.first_forward = False
        # sigmoid here
        return self.softmax(self.A3R)

    def backward(self, Y, learning_rate):
        if self.first_backward:
            print("\nBackward Propagation:")
            print(f"{'Layer':<15}{'Operation':<20}{'Computation':<30}{'FLOP backward':<20}")
            print("-" * 85)

        # Output Layer
        delta3R = self.A3R - Y
        if self.first_backward:
            flops = delta3R.size
            print(f"{'Loss':<15}{'δ3R = A3R - Y':<20}{'output*batch':<30}{flops:<20}")
            self.backward_flops += flops

        delta3 = delta3R  * self.relu_derivative(self.A3)
        if self.first_backward:
            flops = delta3.size
            print(f"{'RELU':<15}{'δ3 = dδ3R/dA3':<20}{'output*batch':<30}{flops:<20}")
            self.backward_flops += flops
        dW3 = np.dot(self.A2R.T, delta3)
        if self.first_backward:
            flops = 2 * self.W2.shape[1] * self.W3.shape[1] * self.batch_size
            print(f"{'Hidden2':<15}{'dL/dW3 = A2R.T @ δ3':<20}{'2*hidden2*output*batch':<30}{flops:<20}")
            self.backward_flops += flops

        # Hidden2 Layer

        delta2R = np.dot(delta3, self.W3.T)
        if self.first_backward:
            flops = 2 * self.W2.shape[1] * self.W3.shape[1] * self.batch_size
            print(f"{'Derivative':<15}{'δ2R = (δ3@W3.T)':<20}{'2*hidden2*output*batch':<30}{flops:<20}")
            self.backward_flops += flops
        delta2 = delta2R * self.relu_derivative(self.A2)
        if self.first_backward:
            flops = delta2.size
            print(f"{'RELU':<15}{'δ2 = dδ2R/dA2':<20}{'hidden2*batch':<30}{flops:<20}")
            self.backward_flops += flops

        dW2 = np.dot(self.A1R.T, delta2)
        if self.first_backward:
            flops = 2 * self.W1.shape[1] * self.W2.shape[1] * self.batch_size
            print(f"{'Hidden1':<15}{'dL/dW2 = A1R.T @ δ2':<20}{'2*hidden1*hidden2*batch':<30}{flops:<20}")
            self.backward_flops += flops

        # Hidden1 Layer
        delta1R = np.dot(delta2, self.W2.T)
        if self.first_backward:
            flops = 2 * self.W1.shape[1] * self.W2.shape[1] * self.batch_size
            print(f"{'Derivative':<15}{'δ1R = (δ2@W2.T)':<20}{'2*hidden1*hidden2*batch':<30}{flops:<20}")
            self.backward_flops += flops

        delta1 = delta1R * self.relu_derivative(self.A1)
        if self.first_backward:
            flops = delta1.size
            print(f"{'RELU':<15}{'δ1 = dδ1R/dA1':<20}{'hidden1*batch':<30}{flops:<20}")
            self.backward_flops += flops
        dW1 = np.dot(self.X.T, delta1)
        if self.first_backward:
            flops = 2 * self.X.shape[1] * self.W1.shape[1] * self.batch_size
            print(f"{'Input':<15}{'dL/dW1 = X.T @ δ1':<20}{'2*input*hidden1*batch':<30}{flops:<20}")
            self.backward_flops += flops

        # Update weights
        self.W1 -= learning_rate * dW1
        self.W2 -= learning_rate * dW2
        self.W3 -= learning_rate * dW3
        self.b1 -= learning_rate * np.sum(delta1, axis=0, keepdims=True)
        self.b2 -= learning_rate * np.sum(delta2, axis=0, keepdims=True)
        self.b3 -= learning_rate * np.sum(delta3, axis=0, keepdims=True)

        if self.first_backward:
            total_weights = self.W1.size + self.W2.size + self.W3.size + self.b1.size + self.b2.size + self.b3.size
            print(f"{'Update':<15}{'W+=lr*δW':<20}{'2*weights':<30}{2 * total_weights:<20}")
            self.backward_flops += flops
            print("-" * 50)
            print(f"Forward Flops: {self.forward_flops}")
            print(f"Backward Flops: {self.backward_flops}")

        self.first_backward = False

    def train(self, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(Y, learning_rate)

            loss = -np.mean(Y * np.log(output + 1e-8))
            if epoch % 1000 == 0:
               print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


def create_spiral_data(points_per_class, classes):
    X = np.zeros((points_per_class*classes, 2))
    y = np.zeros(points_per_class*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points_per_class*class_number, points_per_class*(class_number+1))
        r = np.linspace(0.0, 1, points_per_class)
        t = np.linspace(class_number*4, (class_number+1)*4, points_per_class) + np.random.randn(points_per_class)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

# 创建螺旋数据集
X, y = create_spiral_data(500, 2)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 转换标签为one-hot编码
Y_train = np.eye(2)[y_train]
Y_test = np.eye(2)[y_test]

# 创建并训练神经网络
nn = ThreeLayerNN(input_size=2, hidden1_size=20, hidden2_size=20, output_size=2)
nn.train(X_train, Y_train, epochs=10000, learning_rate=0.001)

# 在训练集上预测
train_predictions = nn.predict(X_train)
train_accuracy = np.mean(train_predictions == y_train)

# 在测试集上预测
test_predictions = nn.predict(X_test)
test_accuracy = np.mean(test_predictions == y_test)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# 可视化结果
plt.figure(figsize=(15, 5))

# 训练集结果
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolor='black')
plt.title("Training Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 创建网格来可视化决策边界
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)

# 测试集结果
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolor='black')
plt.title("Test Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)

plt.tight_layout()
plt.show()
