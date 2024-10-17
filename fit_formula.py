# Import necessary libraries
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Given data points (pixel heights and corresponding distances)
h_values = np.array([20, 200, 400])
d_values = np.array([5, 20, 60])

# Define the models
# Power Law Model
def power_law(h, a, b):
    return a * h ** b

# Quadratic Model
def quadratic(h, a, b, c):
    return a * h**2 + b * h + c

# Exponential Model
def exponential(h, a, b, c):
    return a * np.exp(b * h) + c

# Fit each model to the data
params_power, _ = curve_fit(power_law, h_values, d_values)
params_quad, _ = curve_fit(quadratic, h_values, d_values)
params_exp, _ = curve_fit(exponential, h_values, d_values)

# Generate a range of pixel heights for plotting
h_plot = np.linspace(0, 400, 100)
d_power = power_law(h_plot, *params_power)
d_quad = quadratic(h_plot, *params_quad)
d_exp = exponential(h_plot, *params_exp)

# Plot the data and the fitted models
plt.figure(figsize=(10, 6))
plt.plot(h_plot, d_power, label=f'Power Law Fit: a={params_power[0]:.4f}, b={params_power[1]:.2f}', color='b')
plt.plot(h_plot, d_quad, label=f'Quadratic Fit', color='g', linestyle='--')
# plt.plot(h_plot, d_exp, label=f'Exponential Fit', color='c', linestyle=':')
plt.scatter(h_values, d_values, color='r', marker='o', label='Data Points')
plt.xlabel('Pixel Height from Bottom of Frame (h)')
plt.ylabel('Distance to Ground (d) [meters]')
plt.title('Comparison of Power Law, Quadratic, and Exponential Fits')
plt.legend()
plt.grid(True)
plt.show()


# calculate the hieght change
# hieght of pixel shift = Hieght of object * focal length / Distance to object

# 