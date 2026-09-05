import numpy as np

x = np.array([1., 2., 3., 4.])
y = np.array([5., 8., 11., 14.])

w = 0.0
b = 0.0

lr = 0.01 #learning rate
epochs = 1000 #number of iterations

for epoch in range(epochs):
    # Step 1：Forward
    y_hat = w * x + b # prediction
    error = y_hat - y # prediction error

    # Step 2：计算 MSE Loss
    loss = np.mean(error ** 2) # MSE(mean squared error)

    # Step 3：计算 Gradient
    gradient_w =2 * error @ x.T / len(x)  #也可以是 np.mean(2 * error * x)
    gradient_b = np.mean(2 * error)

    # Step 4：Gradient Descent
    w = w - lr * gradient_w
    b = b - lr * gradient_b

    # Step 5：观察 Training
    if (epoch+1) % 10 == 0:
        print(
        f"epoch={epoch}, "
        f"w={w:.4f}, "
        f"b={b:.4f}, "
        f"loss={loss:.6f}"
    )