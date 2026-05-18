import numpy as np
import matplotlib.pyplot as plt

ul = [1, 0]
ur = [0.7, 0]
CFL = 0.9
gamma = 2
t_final = 1
N = 1000
x = np.linspace(-3, 4, N)
dx = x[1] - x[0]
d = np.where(x < 0, ul[0], ur[0])
u = np.where(x < 0, ul[1], ur[1])
U = np.vstack((d, u))
def flux(U):
    d = U[0]
    u = U[1]
    p = d**(-gamma)
    return np.vstack((-u, p))
def max_speed(U):
    d = U[0]
    return np.max(np.sqrt(gamma) * d**(-(gamma + 1) / 2))

def sound_speed(d):
    return np.sqrt(gamma) * d**(-(gamma + 1) / 2)


def hll_flux(UL, UR):
    dL = UL[0]
    dR = UR[0]

    cL = sound_speed(dL)
    cR = sound_speed(dR)

    SL = np.minimum(-cL, -cR)
    SR = np.maximum(cL, cR)

    FL = flux(UL)
    FR = flux(UR)

    FHLL = np.zeros_like(FL)

    # Case 1: all waves move right
    maskL = SL >= 0.0
    FHLL[:, maskL] = FL[:, maskL]

    # Case 2: all waves move left
    maskR = SR <= 0.0
    FHLL[:, maskR] = FR[:, maskR]

    # Case 3: star region crosses interface
    maskM = ~(maskL | maskR)
    FHLL[:, maskM] = (
        SR[maskM] * FL[:, maskM]
        - SL[maskM] * FR[:, maskM]
        + SL[maskM] * SR[maskM] * (UR[:, maskM] - UL[:, maskM])
    ) / (SR[maskM] - SL[maskM])

    return FHLL


t = 0.0
while t < t_final:
    dt = CFL * dx / max_speed(U)   # fixed line
    if t + dt > t_final:
        dt = t_final - t

    UL = U[:, :-1]
    UR = U[:, 1:]

    F_half = hll_flux(UL, UR)
    U_new = U.copy()
    U_new[:, 1:-1] = U[:, 1:-1] - (dt / dx) * (F_half[:, 1:] - F_half[:, :-1])

    # zero-gradient boundary conditions
    U_new[:, 0] = U_new[:, 1]
    U_new[:, -1] = U_new[:, -2]
    U = U_new
    # optional safety check
    if np.any(U[0] <= 0):
        raise ValueError("Nonphysical state: d became non-positive.")

    t += dt


plt.figure()
plt.plot(x, U[0], label='HLL')
plt.xlabel('x')
plt.ylabel('d')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(x, U[1], label='HLL')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)

plt.show()