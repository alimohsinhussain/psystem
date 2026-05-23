import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
gamma = 2.0
CFL = 0.9
t_final = 1.0

ul = [1.0, 0.0]
ur = [0.7, 0.0]

N = 1000
x = np.linspace(-3.0, 4.0, N)
dx = x[1] - x[0]

def flux(U):
    d = U[0]
    u = U[1]
    p = d**(-gamma)
    return np.vstack((-u, p))

def sound_speed(d):
    d_safe = np.maximum(d, 1e-12)
    return np.sqrt(gamma) * d_safe**(-(gamma + 1.0) / 2.0)

def max_speed(U):
    return np.max(sound_speed(U[0]))

def initial_condition(x):
    d = np.where(x < 0.0, ul[0], ur[0])
    u = np.where(x < 0.0, ul[1], ur[1])
    return np.vstack((d, u))


# Exact Riemann solution at time t = 1

def exact_solution(x, t):
    dL = 1.0
    uL = 0.0

    dR = 0.7
    uR = 0.0

    d_star = 0.83
    u_star = -0.27

    s1 = -1.63

    lambda2_star = np.sqrt(gamma) * d_star**(-(gamma + 1.0) / 2.0)
    lambda2_R = np.sqrt(gamma) * dR**(-(gamma + 1.0) / 2.0)

    xi = x / t

    d_exact = np.zeros_like(x)
    u_exact = np.zeros_like(x)

    # left state
    mask1 = xi < s1
    d_exact[mask1] = dL
    u_exact[mask1] = uL

    # intermediate state
    mask2 = (xi >= s1) & (xi < lambda2_star)
    d_exact[mask2] = d_star
    u_exact[mask2] = u_star

    # 2-rarefaction fan
    mask3 = (xi >= lambda2_star) & (xi <= lambda2_R)

    d_exact[mask3] = (np.sqrt(gamma) / xi[mask3])**(2.0 / 3.0)

    d_fan = d_exact[mask3]
    u_exact[mask3] = 2.0 * np.sqrt(gamma) * (
        1.0 / np.sqrt(d_fan) - 1.0 / np.sqrt(dR)
    )

    # right state
    mask4 = xi > lambda2_R
    d_exact[mask4] = dR
    u_exact[mask4] = uR

    return np.vstack((d_exact, u_exact))


# Lax-Friedrichs numerical flux

def lax_friedrichs_flux(UL, UR, dx, dt):
    FL = flux(UL)
    FR = flux(UR)

    return 0.5 * (FL + FR) - 0.5 * (dx / dt) * (UR - UL)

# HLL numerical flux

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
    FHLL[:, maskL] = FL[:, maskL]

    maskR = SR <= 0.0
    FHLL[:, maskR] = FR[:, maskR]

    maskM = ~(maskL | maskR)
    FHLL[:, maskM] = (
        SR[maskM] * FL[:, maskM]
        - SL[maskM] * FR[:, maskM]
        + SL[maskM] * SR[maskM] * (UR[:, maskM] - UL[:, maskM])
    ) / (SR[maskM] - SL[maskM])

    return FHLL


# General finite volume solver

def solve(method):
    U = initial_condition(x)
    t = 0.0

    while t < t_final:
        dt = CFL * dx / max_speed(U)

        if t + dt > t_final:
            dt = t_final - t

        UL = U[:, :-1]
        UR = U[:, 1:]

        if method == "LF":
            F_half = lax_friedrichs_flux(UL, UR, dx, dt)

        elif method == "HLL":
            F_half = hll_flux(UL, UR)

        else:
            raise ValueError("Unknown method. Use 'LF' or 'HLL'.")

        U_new = U.copy()

        U_new[:, 1:-1] = (
            U[:, 1:-1]
            - (dt / dx) * (F_half[:, 1:] - F_half[:, :-1])
        )

        # Zero-gradient boundary conditions
        U_new[:, 0] = U_new[:, 1]
        U_new[:, -1] = U_new[:, -2]

        U = U_new

        if np.any(U[0] <= 0.0):
            raise ValueError(f"Nonphysical state in {method}: d became non-positive.")

        t += dt

    return U


# Compute solutions

U_LF = solve("LF")
U_HLL = solve("HLL")
U_exact = exact_solution(x, t_final)

d_LF, u_LF = U_LF[0], U_LF[1]
d_HLL, u_HLL = U_HLL[0], U_HLL[1]
d_exact, u_exact = U_exact[0], U_exact[1]





# Plot d comparison

plt.figure(figsize=(9, 4))
plt.plot(x, d_exact, 'green', linewidth=2.0, label="Exact")
plt.plot(x, d_LF, 'blue', linewidth=1.6, label="Lax-Friedrichs")
plt.plot(x, d_HLL, 'red', linewidth=1.6, label="Godunov using HLL flux")
plt.xlabel(r"$x$")
plt.ylabel(r"$d(x,1)$")
plt.title(r"$d(x,1)$")
plt.grid(True)
plt.legend()
plt.tight_layout()


# Plot u comparison

plt.figure(figsize=(9, 4))
plt.plot(x, u_exact, 'green', linewidth=2.0, label="Exact")
plt.plot(x, u_LF, 'blue', linewidth=1.6, label="Lax-Friedrichs")
plt.plot(x, u_HLL, 'red', linewidth=1.6, label="Godunov using HLL flux")
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x,1)$")
plt.title(r"$u(x,1)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()







