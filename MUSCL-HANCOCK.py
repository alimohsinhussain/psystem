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
d_floor = 1e-12

def flux(U):
    d = U[0]
    u = U[1]
    p = d ** (-gamma)
    return np.vstack((-u, p))

def sound_speed(d):
    return np.sqrt(gamma) * d ** (-(gamma + 1) / 2)

def max_speed(U):
    return np.max(sound_speed(U[0]))

def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

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

    maskL = SL >= 0.0
    maskR = SR <= 0.0
    maskM = ~(maskL | maskR)

    FHLL[:, maskL] = FL[:, maskL]
    FHLL[:, maskR] = FR[:, maskR]
    FHLL[:, maskM] = (
        SR[maskM] * FL[:, maskM]
        - SL[maskM] * FR[:, maskM]
        + SL[maskM] * SR[maskM] * (UR[:, maskM] - UL[:, maskM])
    ) / (SR[maskM] - SL[maskM])

    return FHLL

def solve_muscl_hancock(N):
    x = np.linspace(-3.0, 4.0, N)
    dx = x[1] - x[0]

    d = np.where(x < 0, ul[0], ur[0])
    u = np.where(x < 0, ul[1], ur[1])
    U = np.vstack((d, u))

    t = 0.0

    while t < t_final:
        dt = CFL * dx / max_speed(U)

        if t + dt > t_final:
            dt = t_final - t

        # Limited slopes
        delta = np.zeros_like(U)
        delta_left = U[:, 1:-1] - U[:, :-2]
        delta_right = U[:, 2:] - U[:, 1:-1]
        delta[:, 1:-1] = minmod(delta_left, delta_right)

        # Piecewise linear reconstruction
        U_L = U - 0.5 * delta
        U_R = U + 0.5 * delta

        # Positivity floor
        U_L[0] = np.maximum(U_L[0], d_floor)
        U_R[0] = np.maximum(U_R[0], d_floor)

        # Hancock predictor
        F_L = flux(U_L)
        F_R = flux(U_R)

        U_L_bar = U_L + 0.5 * (dt / dx) * (F_L - F_R)
        U_R_bar = U_R + 0.5 * (dt / dx) * (F_L - F_R)

        U_L_bar[0] = np.maximum(U_L_bar[0], d_floor)
        U_R_bar[0] = np.maximum(U_R_bar[0], d_floor)

        # Interface states
        U_L_half = U_R_bar[:, :-1]
        U_R_half = U_L_bar[:, 1:]

        # HLL flux
        F_half = hll_flux(U_L_half, U_R_half)

        # Conservative update
        U_new = U.copy()
        U_new[:, 1:-1] = U[:, 1:-1] - (dt / dx) * (F_half[:, 1:] - F_half[:, :-1])

        # Zero-gradient BC
        U_new[:, 0] = U_new[:, 1]
        U_new[:, -1] = U_new[:, -2]

        if np.any(U_new[0] <= 0):
            raise ValueError("d became non-positive.")

        U = U_new
        t += dt

    return x, U, dx

x, U_num, dx = solve_muscl_hancock(1000)
plt.figure(figsize=(9, 4))
plt.plot(x, U_num[0], label="MUSCL-Hancock + HLL")
plt.xlabel("x")
plt.ylabel("d(x,1)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


plt.figure(figsize=(9, 4))
plt.plot(x, U_num[1], label="MUSCL-Hancock + HLL")
plt.xlabel("x")
plt.ylabel("u(x,1)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()