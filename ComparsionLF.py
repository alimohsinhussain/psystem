import numpy as np
import matplotlib.pyplot as plt

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

#Lax frederick Method
ul = [1, 0]

ur = [0.7, 0]

CFL = 0.9
gamma = 2
t_final = 1
N = 2000

x= np.linspace(-3,4,N)

dx = x[1]-x[0]

print(f"Step Size = {dx}")

v = np.where(x < 0, ul[0], ur[0])
u = np.where(x < 0, ul[1], ur[1])

# VECTOR U
U = np.vstack((v, u))
print(np.shape(U))
# FLUX VECTOR
def flux(U):
  v = U[0]
  u = U[1]
  p = v**(-gamma)
  return np.vstack((-u,p))

def max_speed(v):
    return np.max(np.sqrt(gamma) * v**(-(gamma + 1) / 2))

t = 0

while t<t_final:

  dt = CFL* dx / max_speed(U[0])

  if t + dt > t_final:
          dt = t_final - t

  F = flux(U)

  UL = U[:,:-1]
  UR = U[:,1:]
  FL = F[:,:-1]
  FR = F[:,1:]

  flux_half =  (dx / (2* dt))*(UL - UR) + 0.5* (FR + FL)

  U[:,1:-1] = U[:,1:-1] - (dt/dx) * (flux_half[:,1:] - flux_half[:,:-1])

  U[:, 0] = U[:, 1]
  U[:, -1] = U[:, -2]

  t = t + dt


# Plot
plt.figure(figsize=(9, 5))
plt.plot(x, d_exact, linewidth=2, label="Exact solution")
plt.plot(x, U[0],color = 'blue', label="Lax Frederick Method")
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




# Comparison for velocity 
uL = 0.0
u_star = -0.276542
uR = 0.0
dR = 0.7

s1 = -1.629421
x_fan_left = 1.869290
x_fan_right = 2.414726

# Since t = 1, xi = x
x = np.linspace(-3.0, 4.0, 2000)

# Allocate solution array
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

# Plotting 
plt.figure(figsize=(9, 5))
plt.plot(x, u_exact,color= 'green', linewidth=2)
plt.plot(x, U[1],color='blue', label="Lax frederick method")
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