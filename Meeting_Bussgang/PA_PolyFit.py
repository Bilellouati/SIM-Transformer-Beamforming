import numpy as np
import matplotlib.pyplot as plt
import math 

def hpa_sspa_modif_rapp(vin, Vsat, p, q, G, A, B):
    a0 = np.abs(vin)
    theta = np.angle(vin)
    Am = (G * a0) / ((1 + (G * a0 / Vsat) ** (2 * p)) ** (1 / (2 * p)))
    Bm = (A * (a0 ** q)) / (1 + (a0 / B) ** q)
    vout = Am * np.exp(1j * (theta + Bm))
    return vout

# Define parameters for the PA model
### PA parameters : Modified Rapp model 
IBO = 3
IBOr= 10**(IBO/10)
p = 1.1
q = 4
Vsat = 1.9
G = 16
A = -345
B = 0.17

# Define the input signal amplitude as a Gaussian random variable
rho=np.linspace(0.0,0.5, 100000)
# Define the input signal phase as a random variable
theta = np.linspace(0, 2*np.pi, 100000)

### Power amplifier model
# The input signal to the transfer function
vin = rho * np.exp(1j * theta)

# Compute the output signal using the transfer function
vout = hpa_sspa_modif_rapp(vin, Vsat, p, q, G, A, B)
# Extract the amplitudes and phases of the output signal
a_out = np.abs(vout)
phi_out = np.angle(vout,deg=False) # in degrees


### Fit polynomial to AM/AM characteristic
degree = 15  # You can change the degree of the polynomial
coeffs = np.polyfit(rho, vout * np.exp(-1j * theta), degree)
poly = np.poly1d(coeffs)
s_out_poly = poly(rho)
vout_poly = s_out_poly* np.exp(1j * theta)
phi_out_poly = np.angle(vout_poly,deg=False) # in degrees

# AM/AM plot with polynomial fit
plt.figure()
plt.scatter(rho, a_out, s=10, alpha=0.5, label='PA output')
plt.plot(rho, np.abs(vout_poly), color='red', linewidth=2, label='Polynomial Fit')
plt.xlabel('Input Amplitude $\\rho=|v_{in}|$')
plt.ylabel('Output Amplitude $|v_{out}|$')
plt.title('AM/AM Characteristic with Polynomial Fit')
plt.grid(True)
plt.legend()
plt.show()
