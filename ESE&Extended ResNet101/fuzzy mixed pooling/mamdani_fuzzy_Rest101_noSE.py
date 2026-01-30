import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt

# Define input and output fuzzy range
x_var = np.arange(0, 5.01, 0.1) #input
x_lambda = np.arange(0, 1.01, 0.1) #output

# Membership functions
var_low = fuzz.trimf(x_var, [0, 0, 2.5])
var_middle = fuzz.trimf(x_var, [0, 2.5, 5.0])
var_high = fuzz.trimf(x_var, [2.5, 5.0, 5.0])

lambda_low = fuzz.trimf(x_lambda, [0, 0, 0.5])
lambda_middle = fuzz.trimf(x_lambda, [0, 0.5, 1.0])
lambda_high = fuzz.trimf(x_lambda, [0.5, 1.0, 1.0])


# Whole config
fig_scale_x = 2.0
fig_scale_y = 1.5
fig = plt.figure(figsize=(6.4 * fig_scale_x, 4.8 * fig_scale_y))
row = 2
col = 3

plt.subplot(row, col, 1)
plt.title("Variance")
plt.plot(x_var, var_low, label="low", marker=".")
plt.plot(x_var, var_middle, label="middle", marker=".")
plt.plot(x_var, var_high, label="high", marker=".")

plt.legend(loc="upper left")

plt.subplot(row, col, 2)
plt.title("Lambda")
plt.plot(x_lambda, lambda_low, label="low", marker=".")
plt.plot(x_lambda, lambda_middle, label="middle", marker=".")
plt.plot(x_lambda, lambda_high, label="high", marker=".")
plt.legend(loc="upper left")
plt.show()


def compute_fuzzy_radius(var_scores):

    var_scores = np.where(var_scores > 5.0, 5.0, var_scores)

    for var_score in var_scores:
       
       lambda_centroid = []
       var_low_degree = fuzz.interp_membership(x_var, var_low, var_score)
       var_middle_degree = fuzz.interp_membership(x_var, var_middle, var_score)
       var_high_degree = fuzz.interp_membership(x_var, var_high, var_score)


       low_degree = var_low_degree

       middle_degree = var_middle_degree

       high_degree = var_high_degree

       activation_low = np.fmin(low_degree, lambda_low)
       activation_middle = np.fmin(middle_degree, lambda_middle)
       activation_high = np.fmin(high_degree, lambda_high)

       #Apply fuzzy rule
       aggregated = np.fmax(
           activation_low,
           np.fmax(activation_middle, activation_high))
    
       if aggregated.sum() == 0:
            fallback_value = 0.5  # or any default value you consider neutral
            lambda_centroid.append(fallback_value)
       else:
            lambda_centroid.append(fuzz.defuzz(x_lambda, aggregated, 'centroid'))
            
       #print(lambda_centroid)
       return lambda_centroid


 
