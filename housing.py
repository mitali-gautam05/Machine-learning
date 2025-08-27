#  5D Dataset Visualization (Single Code)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

#  Step 1:Load Sample 5D Data
# You can replace this with your dataset
np.random.seed(42)
df = pd.DataFrame({
    "Feature1": np.random.rand(100) * 100,
    "Feature2": np.random.rand(100) * 50,
    "Feature3": np.random.rand(100) * 75,
    "Feature4": np.random.rand(100) * 10,   # for color
    "Feature5": np.random.rand(100) * 500   # for size
})

#  Step 2: Static 3D Plot (Matplotlib)
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    df["Feature1"], df["Feature2"], df["Feature3"],
    c=df["Feature4"], s=df["Feature5"]/20, cmap="viridis", alpha=0.7
)
plt.colorbar(sc, label="Feature4 (Color)")
ax.set_xlabel("Feature1")
ax.set_ylabel("Feature2")
ax.set_zlabel("Feature3")
plt.title("5D Visualization (3D + Color + Size)")
plt.show()

# Step 3: Interactive 5D Plot (Plotly)
fig = px.scatter_3d(
    df, 
    x="Feature1", y="Feature2", z="Feature3",
    color="Feature4", size="Feature5",
    color_continuous_scale="Viridis",
    opacity=0.7,
    title="Interactive 5D Visualization"
)
fig.show()
