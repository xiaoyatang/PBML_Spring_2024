import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

# def student_t_pdf(x, nu, mu=0, precision=1):
#     numerator = np.exp(-0.5 * (nu + 1) * np.log(1 + (x - mu) ** 2 / (nu * precision)))
#     denominator = np.sqrt(nu * precision * np.pi) * np.exp(np.log(np.abs(x - mu) + np.sqrt(nu * precision)))
#     return numerator / denominator

# # Define the PDF of the standard Gaussian distribution
# def gaussian_pdf(x, mu=0, sigma=1):
#     return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Set mean, precision and DOF
mu = 0
precision = 1
nu_values = [0.1, 1, 10, 100, 1e6]

x = np.linspace(-8, 8, 1000)
plt.figure(figsize=(10, 6))

# Plot standard Gaussian distribution
plt.plot(x, norm.pdf(x, mu, np.sqrt(1)), label='Standard Gaussian (N(0, 1))')
# plt.plot(x, gaussian_pdf(x, mu, np.sqrt(1)), label='Standard Gaussian (N(0, 1))')

# Plot Student's t-distribution for various degrees of freedom
for nu in nu_values:
    plt.plot(x, t.pdf(x, df=nu, loc=mu, scale=1/np.sqrt(precision)), label=f'Student\'s t (ν={nu})')
    # plt.plot(x,student_t_pdf(x, nu, mu, precision), label=f'Student\'s t (ν={nu})')

plt.title('Density Curves of Student\'s t-distribution and Standard Gaussian')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
