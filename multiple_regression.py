# ==============================
# POLYNOMIAL REGRESSION WORKFLOW
# ==============================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# 1. Generate Quadratic Dataset

X = 6 * np.random.rand(200, 1) - 3
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200, 1)

plt.plot(X, y, 'b.')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Generated Dataset")
plt.show()

# 2. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# 3. Linear Regression (Baseline)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Linear Regression R2:", r2_score(y_test, y_pred))

plt.plot(X_train, lr.predict(X_train), color='r', label="Linear Fit")
plt.plot(X, y, "b.", label="Data")
plt.legend()
plt.show()

# 4. Polynomial Regression (Degree 2)

poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_trans = poly.fit_transform(X_train)
X_test_trans = poly.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_trans, y_train.ravel())  # .ravel() to fix warning
y_pred = lr.predict(X_test_trans)
print("Polynomial Regression (deg=2) R2:", r2_score(y_test, y_pred))

X_new = np.linspace(-3, 3, 200).reshape(200, 1)
X_new_poly = poly.transform(X_new)
y_new = lr.predict(X_new_poly)

plt.plot(X_new, y_new, "r-", linewidth=2, label="Polynomial Degree 2")
plt.plot(X_train, y_train, "b.", label='Train')
plt.plot(X_test, y_test, "g.", label='Test')
plt.legend()
plt.show()

# 5. Pipeline for higher degree

def polynomial_regression(degree):
    X_new = np.linspace(-3, 3, 100).reshape(100, 1)

    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()

    model = Pipeline([
        ("poly_features", poly_features),
        ("std_scaler", std_scaler),
        ("lin_reg", lin_reg),
    ])
    model.fit(X, y.ravel())
    y_new = model.predict(X_new)

    plt.plot(X_new, y_new, 'r', label=f"Degree {degree}")
    plt.plot(X_train, y_train, "b.", linewidth=2)
    plt.plot(X_test, y_test, "g.", linewidth=2)
    plt.legend()
    plt.axis([-3, 3, 0, 10])
    plt.show()

# Example: polynomial_regression(5)

# 6. Polynomial with SGD

poly = PolynomialFeatures(degree=2)
X_train_trans = poly.fit_transform(X_train)
X_test_trans = poly.transform(X_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
sgd.fit(X_train_trans, y_train.ravel())

X_new = np.linspace(-2.9, 2.8, 200).reshape(200, 1)
X_new_poly = poly.transform(X_new)
y_new = sgd.predict(X_new_poly)
y_pred = sgd.predict(X_test_trans)

plt.plot(X_new, y_new, "r-", linewidth=2, label=f"SGD Predictions R2={r2_score(y_test,y_pred):.2f}")
plt.plot(X_train, y_train, "b.", label='Train')
plt.plot(X_test, y_test, "g.", label='Test')
plt.legend()
plt.show()


# 7. 3D Polynomial Regression

x = 7 * np.random.rand(100, 1) - 2.8
y2 = 7 * np.random.rand(100, 1) - 2.8
z = x**2 + y2**2 + 0.2*x + 0.2*y2 + 0.1*x*y2 + 2 + np.random.randn(100, 1)

# Scatter 3D
fig = px.scatter_3d(x=x.ravel(), y=y2.ravel(), z=z.ravel(), title="3D Data")
fig.show()

# Linear regression in 3D
X_multi = np.hstack((x, y2))  # shape (100,2)
lr = LinearRegression()
lr.fit(X_multi, z)

# Create meshgrid for surface
x_input = np.linspace(x.min(), x.max(), 10)
y_input = np.linspace(y2.min(), y2.max(), 10)
xGrid, yGrid = np.meshgrid(x_input, y_input)

final = np.c_[xGrid.ravel(), yGrid.ravel()]
z_final = lr.predict(final).reshape(10, 10)

fig = px.scatter_3d(x=x.ravel(), y=y2.ravel(), z=z.ravel(), title="3D Regression Surface")
fig.add_trace(go.Surface(x=x_input, y=y_input, z=z_final))
fig.show()


# 8. High-degree Polynomial in 3D

poly = PolynomialFeatures(degree=5)  # use smaller degree to avoid overflow
X_multi_trans = poly.fit_transform(X_multi)

lr = LinearRegression()
lr.fit(X_multi_trans, z)

X_test_multi = poly.transform(final)
z_final = lr.predict(X_test_multi).reshape(10, 10)

fig = px.scatter_3d(x=x.ravel(), y=y2.ravel(), z=z.ravel(), title="3D Polynomial Regression (deg=5)")
fig.add_trace(go.Surface(x=x_input, y=y_input, z=z_final))
fig.update_layout(scene=dict(zaxis=dict(range=[0, 50])))
fig.show()
