import numpy as np

x = np.array([1., 2., 3., 4., 5., 6.])
y = np.array([0., 0., 0., 1., 1., 1.])

w = 0.0
b = 0.0

lr = 0.1
epochs = 1000
for epoch in range(epochs):
    # Step 1：Linear score
    z = w * x + b

    # Step 2：Sigmoid
    p = 1 / (1 + np.exp(-z))  #sigmoid function归一

    # Step 3：BCE Loss
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) #binary cross entropy loss 二元交叉熵

    # Step 4：Gradient
    gradient_w = (p - y) @ x.T / len(x) #也可以是 np.mean((p - y) * x)
    gradient_b = np.mean(p - y)

    # Step 5：Update
    w = w - lr * gradient_w
    b = b - lr * gradient_b

# Step 6：训练完后，重新forward，得到新的预测值
z = w * x + b
p = 1 / (1 + np.exp(-z))

y_pred = (p >= 0.5).astype(int) #将概率值转化为0或1的预测值

print("w =", w)
print("b =", b)
print("probability =", p)
print("prediction =", y_pred)
print("true label =", y)