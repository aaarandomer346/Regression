# same as linear regression model
# however, there are multiple variables
# first start with the second degree
# m1x^2 + m2x + b
# m1 and m2 are different values for X
# one key difference is that for every weight, the derivitives will need to be calculated seperately.

import matplotlib.pyplot as plt
import numpy as np
import random

def show_graph():
    plt.show()

def make_graph(x_data, y_data, y, weights, intercept):
    plt.figure(figsize= (10, 5))

    plt.scatter(x_data, y_data, color="Orange")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.plot(
        x_data, 
        y,
        label=f"Custom Regression: y = {weights} + {intercept}"
    )

    plt.title("Custom linear regression visualisation", fontsize=14, fontweight="bold")
    plt.xlabel("X (iv)", fontsize=12)
    plt.ylabel("Y (dv)", fontsize=12)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)

def predict(weights, x, intercept):
    prediction = 0
    for i in range(len(weights)):
        prediction += weights[i] * x ** (len(weights) - i)
    return prediction + intercept

def learn(x, y, weights, intercept, step, learn_amount):
    count = 0
    for a in range(learn_amount):
        count += 1
        # for each weight i need to do the gradient decsent.
        grad_intercept = 0
        weight_slopes = [0] * len(weights)
        # iterate over all values of x
        for i in range(len(x)):
            y_pred = predict(weights, x[i], intercept) # predict
            residual = y_pred - y[i] # find loss
            grad_intercept += 2 * residual # intercept gradient decsent

            for j in range(len(weights)):
                weight_slopes[j] += 2 * (x[i] ** (len(weights) - j)) * residual # gradient descent for all weights in the array
        
        for k in range(len(weights)):
            weight_slopes[k] /= len(x) # average out the gradient descent of all weights
            weights[k] -= weight_slopes[k] * step # adjust all weights

        grad_intercept = grad_intercept / len(x) # average gradient descent for interceopt
        intercept -= step * grad_intercept # adjust intercept]

        if count % 10000 == 0:
            loss = np.mean(residual ** 2)
            print(f"loss: {loss}    |    gradient slopes: {weight_slopes}    |    gradient intercept: {grad_intercept}    |")

    return weights, intercept

def make_r_square(x_data, weights, intercept, y):
    average = 0
    ssres = 0
    sstot = 0

    for k in y:
        average += k
    average /= len(y)

    for i in range(len(x_data)):
        y_predict = predict(weights, x_data[i], intercept)
        ssres += (y_predict - y[i]) ** 2
        sstot += (y[i] - average) ** 2
    
    return 1 - (ssres / sstot)

nth_start = input("starting degree:  ")
nth = input("ending degree: ")

results = []

for i in range(int(nth) - int(nth_start) + 1):
    results.append([None, None, None, None])
# [[degree of polynomial, r^2], [degree of polynomial, r^2]]

x = np.arange(1, 499)
y = np.array([np.float64(516.5), np.float64(528.77), np.float64(533.18), np.float64(558.11), np.float64(562.11), np.float64(578.55), np.float64(593.2), np.float64(585.44), np.float64(582.13), np.float64(567.39), np.float64(560.36), np.float64(549.66), np.float64(568.4), np.float64(546.77), np.float64(555.31), np.float64(558.48), np.float64(574.92), np.float64(592.1), np.float64(589.9), np.float64(591.78), np.float64(574.28), np.float64(561.28), np.float64(531.94), np.float64(519.4), np.float64(542.36), np.float64(536.02), np.float64(542.35), np.float64(534.87), np.float64(525.77), np.float64(520.91), np.float64(535.65), np.float64(539.95), np.float64(554.37), np.float64(549.1), np.float64(550.16), np.float64(542.17), np.float64(545.08), np.float64(570.49), np.float64(571.6), np.float64(578.72), np.float64(572.71), np.float64(570.31), np.float64(581.63), np.float64(577.24), np.float64(582.74), np.float64(561.34), np.float64(545.26), np.float64(533.34), np.float64(520.04), np.float64(529.32), np.float64(524.61), np.float64(518.14), np.float64(503.49), np.float64(504.6), np.float64(489.49), np.float64(498.97), np.float64(501.32), np.float64(506.73), np.float64(503.68), np.float64(502.8), np.float64(517.68), np.float64(512.41), np.float64(514.54), np.float64(496.89), np.float64(484.27), np.float64(462.19), np.float64(484.46), np.float64(491.99), np.float64(502.94), np.float64(497.07), np.float64(499.34), np.float64(505.76), np.float64(510.19), np.float64(501.18), np.float64(504.79), np.float64(492.08), np.float64(469.49), np.float64(467.32), np.float64(481.73), np.float64(469.44), np.float64(483.9), np.float64(505.94), np.float64(493.7), np.float64(482.93), np.float64(486.82), np.float64(479.01), np.float64(486.08), np.float64(492.36), np.float64(488.52), np.float64(479.12), np.float64(491.97), np.float64(479.07), np.float64(473.44), np.float64(467.76), np.float64(466.5), np.float64(481.03), np.float64(457.15), np.float64(471.39), np.float64(483.64), np.float64(484.24), np.float64(490.11), np.float64(491.88), np.float64(495.93), np.float64(495.28), np.float64(502.54), np.float64(506.22), np.float64(507.47), np.float64(520.6), np.float64(524.89), np.float64(530.89), np.float64(530.52), np.float64(521.02), np.float64(518.6), np.float64(521.25), np.float64(512.64), np.float64(500.21), np.float64(492.67), np.float64(477.91), np.float64(477.12), np.float64(472.46), np.float64(483.36), np.float64(472.51), np.float64(462.31), np.float64(466.22), np.float64(470.74), np.float64(482.75), np.float64(501.0), np.float64(477.26), np.float64(479.96), np.float64(468.92), np.float64(457.38), np.float64(427.72), np.float64(415.99), np.float64(443.36), np.float64(445.69), np.float64(451.74), np.float64(449.14), np.float64(456.96), np.float64(443.64), np.float64(430.24), np.float64(431.82), np.float64(415.52), np.float64(399.0), np.float64(390.75), np.float64(404.91), np.float64(410.03), np.float64(418.88), np.float64(398.39), np.float64(377.58), np.float64(397.92), np.float64(409.94), np.float64(404.26),np.float64(410.82), np.float64(394.29), np.float64(381.72), np.float64(379.95), np.float64(381.44), np.float64(368.55), np.float64(366.15), np.float64(362.58), np.float64(372.32), np.float64(385.3), np.float64(379.75), np.float64(368.6), np.float64(360.98), np.float64(356.18), np.float64(339.05), np.float64(355.52), np.float64(364.08), np.float64(378.48), np.float64(388.78),np.float64(382.24), np.float64(370.95), np.float64(378.86), np.float64(378.48), np.float64(382.67), np.float64(388.55), np.float64(388.55), np.float64(404.17), np.float64(403.75), np.float64(403.65), np.float64(418.24), np.float64(416.64), np.float64(403.78), np.float64(411.18), np.float64(407.2), np.float64(395.53), np.float64(402.79), np.float64(403.64), np.float64(382.72), np.float64(364.65), np.float64(375.56), np.float64(369.49), np.float64(381.63), np.float64(401.6), np.float64(403.02), np.float64(406.63), np.float64(419.91), np.float64(417.82), np.float64(418.2), np.float64(417.4), np.float64(400.56), np.float64(410.85), np.float64(408.34), np.float64(398.56), np.float64(397.47), np.float64(401.6), np.float64(383.95), np.float64(389.88), np.float64(372.28), np.float64(364.55), np.float64(391.5), np.float64(381.15), np.float64(369.53), np.float64(374.13), np.float64(385.61), np.float64(386.18), np.float64(373.42), np.float64(379.68), np.float64(381.2), np.float64(381.87), np.float64(385.95), np.float64(384.38), np.float64(382.58), np.float64(387.32), np.float64(381.68), np.float64(383.19), np.float64(379.92), np.float64(380.35), np.float64(382.34), np.float64(393.73), np.float64(404.35), np.float64(401.36), np.float64(397.8), np.float64(384.33), np.float64(382.06), np.float64(380.92), np.float64(378.69), np.float64(372.38), np.float64(363.98), np.float64(374.23), np.float64(376.93), np.float64(378.17), np.float64(378.07), np.float64(382.86), np.float64(382.25), np.float64(381.77),np.float64(379.92), np.float64(375.22), np.float64(373.0), np.float64(373.47), np.float64(371.52), np.float64(374.85), np.float64(371.48), np.float64(377.17), np.float64(380.49), np.float64(378.4), np.float64(380.78), np.float64(379.87), np.float64(380.63), np.float64(380.35), np.float64(378.21), np.float64(373.14), np.float64(379.49), np.float64(384.14), np.float64(380.21),np.float64(372.47), np.float64(379.59), np.float64(379.07), np.float64(374.94), np.float64(373.12), np.float64(377.77), np.float64(378.68), np.float64(373.79), np.float64(369.52), np.float64(366.06), np.float64(371.77), np.float64(370.0), np.float64(371.63), np.float64(368.73), np.float64(365.64), np.float64(369.45), np.float64(368.05), np.float64(367.81), np.float64(362.98), np.float64(362.45), np.float64(351.33), np.float64(342.97), np.float64(338.24), np.float64(348.77), np.float64(352.49), np.float64(355.44), np.float64(352.06), np.float64(353.85), np.float64(356.6), np.float64(355.73), np.float64(358.49), np.float64(362.06), np.float64(360.76), np.float64(365.73), np.float64(367.71), np.float64(371.48), np.float64(371.48), np.float64(371.58), np.float64(371.1), np.float64(373.71), np.float64(369.16), np.float64(362.64), np.float64(355.05), np.float64(359.5), np.float64(360.32), np.float64(364.86), np.float64(367.57), np.float64(365.35), np.float64(365.06), np.float64(359.98), np.float64(359.11), np.float64(360.42), np.float64(358.63), np.float64(357.86), np.float64(363.17), np.float64(364.82), np.float64(362.35), np.float64(362.11), np.float64(364.48), np.float64(362.3), np.float64(360.08), np.float64(365.66), np.float64(361.01), np.float64(377.25), np.float64(370.16), np.float64(366.78), np.float64(371.04), np.float64(362.67), np.float64(354.89), np.float64(357.1), np.float64(354.6), np.float64(354.26), np.float64(354.45), np.float64(351.32), np.float64(350.34), np.float64(353.62), np.float64(351.32), np.float64(351.71), np.float64(353.62), np.float64(350.49), np.float64(348.19), np.float64(344.18), np.float64(348.73), np.float64(354.45), np.float64(353.13), np.float64(350.49), np.float64(349.76), np.float64(342.76), np.float64(345.16), np.float64(345.7), np.float64(335.23), np.float64(335.47), np.float64(335.23), np.float64(336.6), np.float64(336.99), np.float64(335.13), np.float64(328.18), np.float64(326.76), np.float64(329.75), np.float64(331.9), np.float64(336.89), np.float64(346.58), np.float64(347.12), np.float64(349.41), np.float64(348.29), np.float64(343.84), np.float64(349.12), np.float64(359.79), np.float64(372.46), np.float64(372.8), np.float64(374.61), np.float64(376.96), np.float64(363.41), np.float64(355.77), np.float64(362.53), np.float64(359.0), np.float64(361.94), np.float64(361.94), np.float64(363.02), np.float64(362.14), np.float64(351.18), np.float64(348.14), np.float64(358.32), np.float64(353.57), np.float64(356.12), np.float64(369.38), np.float64(367.22), np.float64(356.26), np.float64(355.29), np.float64(364.48), np.float64(373.39), np.float64(373.88), np.float64(368.92), np.float64(356.45), np.float64(352.38), np.float64(350.55), np.float64(347.33), np.float64(343.57), np.float64(350.11), np.float64(355.16), np.float64(360.8), np.float64(350.16), np.float64(350.8), np.float64(348.82), np.float64(350.6), np.float64(350.6), np.float64(350.7), np.float64(346.44), np.float64(335.7), np.float64(325.59), np.float64(338.57), np.float64(335.89), np.float64(321.29), np.float64(331.19), np.float64(329.85), np.float64(340.2), np.float64(328.32), np.float64(329.26), np.float64(334.46), np.float64(344.71), np.float64(330.2), np.float64(330.5), np.float64(316.45), np.float64(320.8), np.float64(327.7), np.float64(318.45), np.float64(305.1), np.float64(322.1), np.float64(324.8), np.float64(327.55), np.float64(329.35), np.float64(342.65), np.float64(346.8), np.float64(349.8), np.float64(347.0), np.float64(352.65), np.float64(364.2), np.float64(373.3), np.float64(382.7), np.float64(383.35), np.float64(383.35), np.float64(376.45), np.float64(374.0), np.float64(378.05), np.float64(375.8), np.float64(363.55), np.float64(354.3), np.float64(354.3), np.float64(353.5), np.float64(356.4), np.float64(359.1), np.float64(359.75), np.float64(359.4), np.float64(351.55), np.float64(344.35), np.float64(346.55), np.float64(348.3), np.float64(345.85), np.float64(333.7), np.float64(334.25), np.float64(333.0), np.float64(333.7), np.float64(336.95), np.float64(342.2), np.float64(337.6), np.float64(339.7), np.float64(339.7), np.float64(337.65), np.float64(331.4), np.float64(325.05), np.float64(325.65), np.float64(335.2), np.float64(343.9), np.float64(343.4), np.float64(352.1), np.float64(348.8), np.float64(336.85), np.float64(348.05), np.float64(351.85), np.float64(356.55), np.float64(358.75), np.float64(358.7), np.float64(355.35), np.float64(369.85), np.float64(361.9), np.float64(363.65), np.float64(364.65), np.float64(364.65), np.float64(364.95), np.float64(363.8), np.float64(370.1), np.float64(369.6), np.float64(365.15)])

#plt.scatter(x, y)
#plt.show()

x_data = x[:300]
y_data = y[:300]

x_test = x[300:499]
y_test = y[300:499]

x_min, x_max = np.min(x_data), np.max(x_data)
y_min, y_max = np.min(y_data), np.max(y_data)

y_data = (y_data - y_min) / (y_max - y_min)
y_test = (y_test - y_min) / (y_max - y_min)

x_data = (x_data - x_min) / (x_max - x_min)
x_test = (x_test - x_min) / (x_max - x_min)

# print(len(x_test, len(x_data))
#print(len(y_test, len(y_data))

for i in range(int(nth_start), int(nth) + 1):
    weights = []
    for j in range(i):
        weights.append(random.uniform(0, 0.1))
    # weights = [0.5, 2, 1]
    intercept = random.uniform(0, 0.1)
    # intercept = 5

    step = 0.001 # * (0.001 ** (i - 2))
    learn_amount = 100000

    weights, intercept = learn(x_data, y_data, weights, intercept, step, learn_amount)
    r_squared = make_r_square(x_test, weights, intercept, y_test)

    results[i - int(nth_start)][0] = i
    results[i - int(nth_start)][1] = r_squared
    results[i - int(nth_start)][2] = weights
    results[i - int(nth_start)][3] = intercept

    plt.close('all')

best_polynomial_fit = None
current_r_squared = None
previous_r_squared = None

for r in results:
    print(r)
    current_r_squared = r[1]
    if previous_r_squared == None:
        previous_r_squared = current_r_squared - 1
    if current_r_squared > previous_r_squared:
        best_polynomial_fit = r[0]
    previous_r_squared = current_r_squared


weights = results[best_polynomial_fit - int(nth_start)][2]
intercept = results[best_polynomial_fit - int(nth_start)][3]

make_graph(x_data, y_data, predict(weights, x_data, intercept), weights, intercept)
show_graph()
make_graph(x_test, y_test, predict(weights, x_test, intercept), weights, intercept)
show_graph()