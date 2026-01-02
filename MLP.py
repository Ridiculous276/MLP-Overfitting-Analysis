'''
A simple example of overfitting using a Multi-Layer Perceptron (MLP) to fit noisy sine wave data.
Author: Dongyang Kuang

NOTE:
    [] Multiple aspects can be investigated:
'''

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# ==============================================================================
# Common Functions & Setup
# ==============================================================================
def sin_2pi_on_grid(x):
    """
    Computes y = sin(2pi*x) on a uniform grid from 0 to 1.

    Parameters:
    x (int or array): input for evaluation.

    Returns:
    y (numpy.ndarray): The computed sine values at the grid points.
    """

    y = np.sin(2 * np.pi * x)  # what if include more periods in [0,1]
    return y


# %%
# Example usage:
num_points = 100  # Are there any sampling method that is more efficient?
x = np.linspace(0, 1, num_points)  # what if non-uniform grid?
y = sin_2pi_on_grid(x)

# Add white noise to y
noise_intensity = 0.4
noise = np.random.normal(0, noise_intensity, len(y))
y_noise = y + noise

# %%
# Create a figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot y vs x on the left subplot
axs[0].plot(x, y, label='y = sin(2πx)')
axs[0].set_title('y vs x')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()

# Plot y_noise vs x on the right subplot
axs[1].plot(x, y_noise, label='y_noise = sin(2πx) + noise', color='orange')
axs[1].set_title('y_noise vs x')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y_noise')
axs[1].legend()

# Display the plots
plt.tight_layout()
plt.show()

# %%
# Define the MLP model
class MLP(nn.Module):
    def __init__(self, hidden_units=32):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(1, hidden_units)  # what if I used different initialization?
        self.hidden2 = nn.Linear(hidden_units, hidden_units)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x


# %%
# Prepare the data
USE_NOISE = True
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
if USE_NOISE:
    y_tensor = torch.tensor(y_noise, dtype=torch.float32).view(-1, 1)
else:
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# %%
# Initialize the model, loss function, and optimizer
model = MLP(hidden_units=32)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Training loop
loss_history = []
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# =======================================================
# [MODIFIED] 绘图部分：加入保存高清图片的逻辑
# =======================================================
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss')

# 添加不可约误差线 (0.4^2 = 0.16)
plt.axhline(y=0.16, color='r', linestyle='--', label='Irreducible Error (0.16)')

plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 【新增】保存为 300 DPI 的高清图片，bbox_inches='tight' 防止边缘被切掉
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
print("Task 1 Figure Saved: 'loss_curve.png'")

plt.show()
# =======================================================

# Evaluate the model
model.eval()
with torch.no_grad():
    predicted = model(x_tensor).numpy()

# %%
# Plot the true values and the predicted values
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='True y = sin(2πx)')
plt.plot(x, predicted, label='Predicted y', linestyle='--')
plt.title('True vs Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# %%
# ==============================================================================
# Task 2: Impact of Noise Intensity (Low vs High) [ADDED SECTION]
# ==============================================================================
print("\n=== Starting Task 2: Noise Intensity Comparison ===")

# Define comparison scenarios
sigmas = [0.1, 0.8]  # Low noise vs High noise
colors = {0.1: 'green', 0.8: 'red'}
results = {}

# Reuse x_tensor from previous section
# Reuse MLP class definition

for s in sigmas:
    print(f"Running experiment with Noise Sigma = {s}...")

    # 1. Generate new noisy data for this sigma
    # (Using the same x and y from the beginning)
    noise_s = np.random.normal(0, s, len(y))
    y_noise_s = y + noise_s
    y_tensor_s = torch.tensor(y_noise_s, dtype=torch.float32).view(-1, 1)

    # 2. Re-initialize Model & Optimizer (Start from scratch)
    model_s = MLP(hidden_units=32)
    optimizer_s = optim.AdamW(model_s.parameters(), lr=0.001)

    # 3. Training Loop
    loss_hist_s = []
    for epoch in range(num_epochs):  # reusing num_epochs=1000
        model_s.train()
        optimizer_s.zero_grad()
        outputs = model_s(x_tensor)
        loss = criterion(outputs, y_tensor_s)
        loss.backward()
        optimizer_s.step()
        loss_hist_s.append(loss.item())

    # 4. Predict
    model_s.eval()
    with torch.no_grad():
        pred_s = model_s(x_tensor).numpy()

    # Store results
    results[s] = {
        'loss': loss_hist_s,
        'pred': pred_s,
        'y_noisy': y_noise_s
    }

# --- Plot Comparison 1: Loss History ---
plt.figure(figsize=(10, 5))
for s in sigmas:
    # [FIXED] Added 'r' before f-string to handle latex backslash \sigma
    plt.plot(results[s]['loss'], label=rf'Training Loss ($\sigma={s}$)', color=colors[s], alpha=0.8)
    # Theoretical irreducible error
    # [FIXED] Added 'r' before f-string
    plt.axhline(y=s ** 2, color=colors[s], linestyle='--', alpha=0.6,
                label=rf'Irreducible Error ($\sigma^2={s ** 2:.2f}$)')

plt.title('Task 2: Loss History Comparison (Low vs High Noise)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('noise_comparison_loss.png', dpi=300, bbox_inches='tight')
print("Task 2 Figure Saved: 'noise_comparison_loss.png'")
plt.show()

# --- Plot Comparison 2: Prediction ---
# [MODIFIED for Overleaf]: Loop through to save separate files instead of subplots
for s in sigmas:
    plt.figure(figsize=(6, 5))  # Single plot size

    # Plot True Signal
    plt.plot(x, y, 'k-', linewidth=2, label='True Signal')

    # Plot Noisy Data
    plt.scatter(x, results[s]['y_noisy'], c=colors[s], alpha=0.3, label=rf'Noisy Data ($\sigma={s}$)')

    # Plot Prediction
    plt.plot(x, results[s]['pred'], 'b--', linewidth=2, label='Prediction')

    plt.title(rf'Prediction ($\sigma={s}$)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Generate filename based on sigma
    suffix = "low" if s == 0.1 else "high"
    filename = f'noise_pred_{suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Task 2 Figure Saved: '{filename}'")
    plt.show()


# ==============================================================================
# Task 3: Extrapolation Capability (New Section)
# ==============================================================================
print("\n=== Starting Task 3: Extrapolation Experiment ===")

# 1. Train a model specifically for this demo (using Low Noise sigma=0.1)
# We re-train to ensure clean demonstration
print("Training model for extrapolation test...")
model_ext = MLP(hidden_units=32)
optimizer_ext = optim.AdamW(model_ext.parameters(), lr=0.001)

# Generate training data for Extrapolation task
y_train_ext = y + np.random.normal(0, 0.1, len(y))
y_tensor_ext = torch.tensor(y_train_ext, dtype=torch.float32).view(-1, 1)

for epoch in range(num_epochs):
    model_ext.train()
    optimizer_ext.zero_grad()
    loss = criterion(model_ext(x_tensor), y_tensor_ext)
    loss.backward()
    optimizer_ext.step()

# 2. Define Extrapolation Range [-1, 2]
# We want to see what happens outside [0, 1]
x_ext = np.linspace(-1, 2, 300)
y_true_ext = sin_2pi_on_grid(x_ext)
x_tensor_ext = torch.tensor(x_ext, dtype=torch.float32).view(-1, 1)

# 3. Predict on extended range
model_ext.eval()
with torch.no_grad():
    pred_ext = model_ext(x_tensor_ext).numpy()

# 4. Plot
plt.figure(figsize=(8, 5))

# Plot True Signal
plt.plot(x_ext, y_true_ext, 'k-', label='True Signal (Periodic)')

# Highlight Training Region [0, 1] with gray background
plt.axvspan(0, 1, color='gray', alpha=0.2, label='Training Region $[0,1]$')

# Plot Prediction
plt.plot(x_ext, pred_ext, 'b--', linewidth=2, label='Model Prediction')

plt.title('Task 3: Extrapolation Failure')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the figure
plt.savefig('extrapolation_demo.png', dpi=300, bbox_inches='tight')
print("Task 3 Figure Saved: 'extrapolation_demo.png'")
plt.show()

# ==============================================================================
# Task 4: Hyperparameter Experiments (Activation, Depth & Width)
# ==============================================================================
print("\n=== Starting Task 4: Hyperparameter Experiments ===")

# Define a flexible MLP class just for Task 4 to allow changing depth/activation
class DynamicMLP(nn.Module):
    def __init__(self, hidden_units=32, num_layers=2, act_type='relu'):
        super(DynamicMLP, self).__init__()
        layers = []

        # 1. Choose Activation Function based on string
        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.ReLU() # Default

        # 2. Build Layers
        # Input Layer (1 -> hidden)
        layers.append(nn.Linear(1, hidden_units))
        layers.append(self.act)

        # Hidden Layers (num_layers - 1)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(self.act)

        # Output Layer (hidden -> 1)
        layers.append(nn.Linear(hidden_units, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Experiment A: Activation Functions (Control Variable: Depth=2, Width=32) ---
print("Running Exp 4.1: Activation Functions...")
activations = ['relu', 'tanh', 'sigmoid']
colors_act = {'relu': 'blue', 'tanh': 'green', 'sigmoid': 'orange'}

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'k-', linewidth=2, label='True Signal', zorder=10)
plt.scatter(x, y_noise, c='gray', alpha=0.3, label='Noisy Data')

for act in activations:
    # Fix depth=2, width=32
    model_act = DynamicMLP(hidden_units=32, num_layers=2, act_type=act)
    opt_act = optim.AdamW(model_act.parameters(), lr=0.005)

    for epoch in range(num_epochs):
        model_act.train()
        opt_act.zero_grad()
        loss = criterion(model_act(x_tensor), y_tensor)
        loss.backward()
        opt_act.step()

    model_act.eval()
    with torch.no_grad():
        pred = model_act(x_tensor).numpy()

    plt.plot(x, pred, color=colors_act[act], linewidth=2, label=f'Activation: {act.upper()}')

plt.title('Task 4.1: Comparison of Activation Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('hyper_activation.png', dpi=300, bbox_inches='tight')
print("Task 4.1 Figure Saved: 'hyper_activation.png'")
plt.show()

# --- Experiment B: Network Depth (Control Variable: Act=ReLU, Width=32) ---
print("Running Exp 4.2: Network Depth...")
depths = [1, 3, 8]
colors_depth = {1: 'orange', 3: 'green', 8: 'purple'}

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'k-', linewidth=2, label='True Signal', zorder=10)
plt.scatter(x, y_noise, c='gray', alpha=0.3, label='Noisy Data')

for d in depths:
    # Fix Act=ReLU, Width=32
    model_depth = DynamicMLP(hidden_units=32, num_layers=d, act_type='relu')
    opt_depth = optim.AdamW(model_depth.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model_depth.train()
        opt_depth.zero_grad()
        loss = criterion(model_depth(x_tensor), y_tensor)
        loss.backward()
        opt_depth.step()

    model_depth.eval()
    with torch.no_grad():
        pred = model_depth(x_tensor).numpy()

    plt.plot(x, pred, color=colors_depth[d], linewidth=2, label=f'Depth: {d} layers')

plt.title('Task 4.2: Comparison of Network Depth')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('hyper_depth.png', dpi=300, bbox_inches='tight')
print("Task 4.2 Figure Saved: 'hyper_depth.png'")
plt.show()

# --- Experiment C: Network Width / Hidden Units (Control Variable: Act=ReLU, Depth=2) ---
# [NEW SECTION ADDED HERE]
print("Running Exp 4.3: Network Width (Hidden Units)...")
widths = [4, 32, 128] # Narrow, Baseline, Wide
colors_width = {4: 'orange', 32: 'green', 128: 'purple'}

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'k-', linewidth=2, label='True Signal', zorder=10)
plt.scatter(x, y_noise, c='gray', alpha=0.3, label='Noisy Data')

for w in widths:
    # Fix Act=ReLU, Depth=2
    model_width = DynamicMLP(hidden_units=w, num_layers=2, act_type='relu')
    opt_width = optim.AdamW(model_width.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model_width.train()
        opt_width.zero_grad()
        loss = criterion(model_width(x_tensor), y_tensor)
        loss.backward()
        opt_width.step()

    model_width.eval()
    with torch.no_grad():
        pred = model_width(x_tensor).numpy()

    plt.plot(x, pred, color=colors_width[w], linewidth=2, label=f'Width: {w} units')

plt.title('Task 4.3: Comparison of Network Width')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('hyper_width.png', dpi=300, bbox_inches='tight')
print("Task 4.3 Figure Saved: 'hyper_width.png'")
plt.show()