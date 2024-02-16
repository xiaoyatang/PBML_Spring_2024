import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

x = np.linspace(0, 1, 1000)

# Define Beta distributions
beta_params1 = (1, 1)
beta_params2 = (5, 5)
beta_params3 = (10, 10)
beta_params4 = (1, 2)
beta_params5 = (5, 6)
beta_params6 = (10, 11)

beta_pdf1 = beta.pdf(x, *beta_params1)
beta_pdf2 = beta.pdf(x, *beta_params2)
beta_pdf3 = beta.pdf(x, *beta_params3)
beta_pdf4 = beta.pdf(x, *beta_params4)
beta_pdf5 = beta.pdf(x, *beta_params5)
beta_pdf6 = beta.pdf(x, *beta_params6)

# Plot density curves for Beta(1,1), Beta(5,5), and Beta(10,10)
plt.figure(figsize=(10, 6))
plt.plot(x, beta_pdf1, label='Beta(1, 1)')
plt.plot(x, beta_pdf2, label='Beta(5, 5)')
plt.plot(x, beta_pdf3, label='Beta(10, 10)')
plt.title('Density Curves for Beta(1,1), Beta(5,5), and Beta(10,10)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Plot density curves for Beta(1,2), Beta(5,6), and Beta(10,11)
plt.figure(figsize=(10, 6))
plt.plot(x, beta_pdf4, label='Beta(1, 2)')
plt.plot(x, beta_pdf5, label='Beta(5, 6)')
plt.plot(x, beta_pdf6, label='Beta(10, 11)')
plt.title('Density Curves for Beta(1,2), Beta(5,6), and Beta(10,11)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
