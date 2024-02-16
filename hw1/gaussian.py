import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy as sci
from scipy.stats import t,norm

np.random.seed(666) 
samples = np.random.normal(loc=0, scale=2, size=30)
print(samples)

def log_likelihood_gaussian(params, data):
    mu, sigma = params
    return -np.sum(norm.logpdf(data, loc=mu, scale=sigma))

def log_likelihood_student_t(params, data):
    mu, sigma, nu = params
    return -np.sum(t.logpdf(data, nu, mu, sigma))

# Define initial values for optimization
initial_params_gaussian = [0, 1]  # Initial guess for mean and standard deviation
initial_params_student_t = [0, 1, 1]  # Initial guess for mean, precision, and degrees of freedom

# Optimize using L-BFGS 
result_gaussian = minimize(log_likelihood_gaussian, initial_params_gaussian, args=(samples,), method='L-BFGS-B')
mu_gaussian, sigma_gaussian = result_gaussian.x
print(mu_gaussian,sigma_gaussian)
result_student_t = minimize(log_likelihood_student_t, initial_params_student_t, args=(samples,), method='L-BFGS-B')
mu_student_t, sigma_student_t, nu_student_t = result_student_t.x
print(mu_student_t,sigma_student_t,nu_student_t)

# Plot density curves and scatter data points for Gaussian distribution
plt.figure(figsize=(10, 6))
plt.hist(samples,density=True,label='Data Points')

x = np.linspace(-8, 8, 1000)
plt.plot(x, norm.pdf(x, 0, 2),label='Sampled Standard Gaussian Density')
plt.plot(x, norm.pdf(x, mu_gaussian, sigma_gaussian), label='Estimated Gaussian Density')
plt.plot(x, t.pdf(x, nu_student_t, mu_student_t, sigma_student_t), label='Estimated Student t Density')

plt.title('MLE of Gaussian and Student t)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

# Inject noise into data
noise = [8, 9, 10]
samples_with_noise = np.concatenate((samples, noise))
print(samples_with_noise)

# Optimize parameters again after injecting noise for Gaussian distribution
result_gaussian_noise = minimize(log_likelihood_gaussian, initial_params_gaussian, args=(samples_with_noise,), method='L-BFGS-B')
mu_gaussian_noise, sigma_gaussian_noise = result_gaussian_noise.x
print(mu_gaussian_noise,sigma_gaussian_noise)

# Optimize parameters again after injecting noise for Student's t distribution
result_student_t_noise = minimize(log_likelihood_student_t, initial_params_student_t, args=(samples_with_noise,), method='L-BFGS-B')
mu_student_t_noise, sigma_student_t_noise, nu_student_t_noise = result_student_t_noise.x
print(mu_student_t_noise,sigma_student_t_noise,nu_student_t_noise)

plt.figure(figsize=(10, 6))
plt.hist(samples_with_noise,density=True,label='Noisy Data Points')

plt.plot(x, norm.pdf(x, mu_gaussian_noise, sigma_gaussian_noise), label='Estimated Gaussian Density')
plt.plot(x, t.pdf(x, nu_student_t_noise, mu_student_t_noise, sigma_student_t_noise), label='Estimated Student t Density')
plt.title('MLE of Gaussian and Student t with Noise')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
