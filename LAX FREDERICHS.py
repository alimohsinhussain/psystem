import numpy as np
import matplotlib.pyplot as plt 
ul = [1, 0]
ur = [0.7, 0]
CFL = 0.9
gamma = 2
t_final = 1
N = 1000
x= np.linspace(-3,4,N)
dx = x[1]-x[0]  
d = np.where(x < 0, ul[0], ur[0])
u = np.where(x < 0, ul[1], ur[1])
U = np.vstack((d, u))

def flux(U):
  d = U[0]
  u = U[1]
  p = d**(-gamma)
  return np.vstack((-u,p))

def max_speed(d):
  return np.max(np.sqrt(gamma)* d**(-(gamma + 1) / 2))

def lax_numerical_flux(UL, UR, dt, dx):
  FL = flux(UL)
  FR = flux(UR)
  alpha = dx / dt
  return 0.5 * (FL + FR) - 0.5 * alpha * (UR - UL)

t = 0
#  Lax frederchs
while t < t_final:
   
  dt = CFL * dx / max_speed(U[0])
  if t + dt > t_final:
    dt = t_final - t

  UL = U[:,:-1]
  UR = U[:,1:]
  F_half = lax_numerical_flux(UL, UR, dt, dx)
  U[:,1:-1] = U[:,1:-1] - (dt/dx) * (F_half[:,1:] - F_half[:,:-1])
  U[:, 0] = U[:, 1]
  U[:, -1] = U[:, -2]

  t += dt

plt.figure()
plt.plot(x, U[0], label= 'lax frederich')
plt.xlabel('x')
plt.ylabel('d')
plt.legend()
plt.grid(True)
plt.figure()
plt.plot(x, U[1], label='lax frederich')
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.show()

