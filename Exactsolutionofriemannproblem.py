import numpy as np
import matplotlib.pyplot as plt

# Exact solution for deformation gradient d
dL = 1.0
d_star = 0.830282
dR = 0.7
s1 = -1.629421
x_fan_left = 1.869290
x_fan_right = 2.414726

x = np.linspace(-3.0, 4.0, 2000)


d_exact = np.zeros_like(x)

# left state
mask1 = x < s1
d_exact[mask1] = dL

# intermediate state
mask2 = (x >= s1) & (x < x_fan_left)
d_exact[mask2] = d_star

# rarefaction fan
mask3 = (x >= x_fan_left) & (x <= x_fan_right)
d_exact[mask3] = (np.sqrt(2.0) / x[mask3])**(2.0 / 3.0)

# right state
mask4 = x > x_fan_right
d_exact[mask4] = dR


plt.figure(figsize=(9, 5))
plt.plot(x, d_exact, linewidth=2, label="Exact solution")
plt.axvline(s1, color='red', linestyle='--', linewidth=1.2, label='1-shock')
plt.axvline(x_fan_left, color='green', linestyle='--', linewidth=1.2, label='2-fan left edge')
plt.axvline(x_fan_right, color='black', linestyle='--', linewidth=1.2, label='2-fan right edge')
plt.xlabel(r'$x$', fontsize=13)
plt.ylabel(r'$d(x,1)$', fontsize=13)


plt.xlim(-3, 4)
plt.ylim(0.65, 1.05)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()




# Exact solution for velocity variable v
uL = 0.0
u_star = -0.276542
uR = 0.0
dR = 0.7

s1 = -1.629421
x_fan_left = 1.869290
x_fan_right = 2.414726

# Since t = 1, xi = x
x = np.linspace(-3.0, 4.0, 2000)


u_exact = np.zeros_like(x)

# left state
mask1 = x < s1
u_exact[mask1] = uL

# intermediate state
mask2 = (x >= s1) & (x < x_fan_left)
u_exact[mask2] = u_star

# rarefaction fan
mask3 = (x >= x_fan_left) & (x <= x_fan_right)

# First compute d(x) inside the fan
d_fan = (np.sqrt(2.0) / x[mask3])**(2.0 / 3.0)

# Then compute u(x) from the 2-rarefaction formula
u_exact[mask3] = 2.0 * np.sqrt(2.0) * (d_fan**(-0.5) - dR**(-0.5))

# right state
mask4 = x > x_fan_right
u_exact[mask4] = uR



# Plot
plt.figure(figsize=(9, 5))
plt.plot(x, u_exact,color= 'green', linewidth=2, label="Exact solution")
plt.axvline(s1, color='red', linestyle='--', linewidth=1.2, label='1-shock')
plt.axvline(x_fan_left, color='green', linestyle='--', linewidth=1.2, label='2-fan left edge')
plt.axvline(x_fan_right, color='black', linestyle='--', linewidth=1.2, label='2-fan right edge')
plt.xlabel(r'$x$', fontsize=13)
plt.ylabel(r'$u(x,1)$', fontsize=13)


plt.xlim(-3, 4)
plt.ylim(-0.35, 0.05)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()