# yep this is what im working on, home hand baked linear regression model. yes im jobless. yes im mentally insane. maybe i understand how this works

# numpy import
# matplotlib import

# y = wx + b (y = weight * x + bias)
# loss is the difference between prediction and actual result, where each difference is squared and then averaged

# then find the gradient of the loss function for both bias and weight find the partial direvative:
# partial direvative of the Loss / partial direvative of the weight ---> for the weight
# partial direvative of the Loss / partial direvative of the bias ---> for the bias

# then take the negative of the 2 values of the weight and bias, multiply by a small step (small float)
# take the change in bias/weight (negative of the partial direvations * a small step) and subtract from the original

import matplotlib.pyplot as plt
import numpy as np
import random as rd

def show_graph():
    plt.show()

def make_graph(x_data, y_data_for_training, y_predict, slope, intercept, r_squared):
    plt.scatter(x_data, y_data_for_training, color="darkorange", edgecolor="black", s=80, label="Actual Data")

    plt.plot(
        x_data,
        y_predict,
        color="navy",
        linestyle="-",
        linewidth=2.5,
        label=f"Custom Regression: y = {slope}x + {intercept}"
    )

    plt.text(
        x=25,                  # X-coordinate on the graph where text starts
        y=60,                 # Y-coordinate on the graph where text starts
        s=f"$R^2 = {r_squared:.3f}$",  # The text (uses $ for LaTeX superscript)
        fontsize=12,          # Size of the font
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')  # Optional box background
    )

    plt.title("Custom linear regression visualisation", fontsize=14, fontweight="bold")
    plt.xlabel("X (iv)", fontsize=12)
    plt.ylabel("Y (dv)", fontsize=12)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)

def predict_y(slope, x, intercept):
    return (slope * x) + intercept

def learn(x_data, slope, intercept, y_data_for_training, step):
    grad_slope = 0
    grad_intercept = 0

    for i in range(len(x_data)):
        y = predict_y(slope, x_data[i], intercept)
        residual = y - y_data_for_training[i]
        grad_slope += 2 * x_data[i] * residual
        grad_intercept += 2 * residual

    grad_slope = grad_slope / len(x_data)
    grad_intercept = grad_intercept / len(x_data)

    slope -= step * grad_slope
    intercept -= step * grad_intercept
    print(grad_intercept)
    print(grad_slope)
    y_predict = predict_y(slope, x_data, intercept)

    return slope, intercept, y_predict

def make_r_square(x_data, slope, intercept, y):
    average = 0
    ssres = 0
    sstot = 0

    for k in y:
        average += k
    average /= len(x_data)

    for i in range(len(x_data)):
        y_predict = predict_y(slope, x_data[i], intercept)
        ssres += (y_predict - y[i]) ** 2
        sstot += (y[i] - average) ** 2
    
    return 1 - (ssres / sstot)

x_data = np.array([
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    22, 35, 42, 48
])
y_data_for_training = np.array([
    42, 45, 43, 49, 52, 51, 56, 55, 59, 62, 60, 
    65, 67, 66, 70, 72, 75, 74, 78, 81, 79, 
    84, 86, 85, 89, 91, 94, 93, 97, 100, 98,
    44, 71, 85, 96
])

slope = rd.uniform(0.1, 3)
intercept = rd.uniform(10, 75)

y_predict = predict_y(slope, x_data, intercept)

make_graph(x_data, y_data_for_training, y_predict, slope, intercept, 0)

plt.figure(figsize=(16, 10))

step = 0.0001 # need to change based on the graph. general rule is higher y values means smaller step, smaller y values larger step
for i in range(250000): # decrease range as see fit
    slope, intercept, y_predict = learn(x_data, slope, intercept, y_data_for_training, step)

r_squared = make_r_square(x_data, slope, intercept, y_data_for_training)

make_graph(x_data, y_data_for_training, y_predict, slope, intercept, r_squared)
show_graph()