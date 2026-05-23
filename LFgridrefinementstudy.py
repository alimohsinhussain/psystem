import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


gamma = 2.0
CFL = 0.9
t_final = 1.0

# Left and right states
ul = [1.0, 0.0]   
ur = [0.7, 0.0]  

# Exact Riemann solution parameters at t = 1
dL = 1.0
d_star = 0.830282
dR = 0.7

uL = 0.0
u_star = -0.276542
uR = 0.0

s1 = -1.629421
x_fan_left = 1.869290
x_fan_right = 2.414726



def exact_solution(x):
    d_exact = np.zeros_like(x)
    u_exact = np.zeros_like(x)

    # left state
    mask1 = x < s1
    d_exact[mask1] = dL
    u_exact[mask1] = uL

    # intermediate constant state
    mask2 = (x >= s1) & (x < x_fan_left)
    d_exact[mask2] = d_star
    u_exact[mask2] = u_star

    # rarefaction fan
    mask3 = (x >= x_fan_left) & (x <= x_fan_right)
    d_exact[mask3] = (np.sqrt(2.0) / x[mask3])**(2.0 / 3.0)

    d_fan = d_exact[mask3]
    u_exact[mask3] = 2.0 * np.sqrt(2.0) * (d_fan**(-0.5) - dR**(-0.5))

    # right state
    mask4 = x > x_fan_right
    d_exact[mask4] = dR
    u_exact[mask4] = uR

    return np.vstack((d_exact, u_exact))

def flux(U):
    d = U[0]
    u = U[1]
    p = d**(-gamma)
    return np.vstack((-u, p))


def max_speed(d):
    return np.max(np.sqrt(gamma) * d**(-(gamma + 1) / 2))


def solve_lf(N):
    x = np.linspace(-3.0, 4.0, N)
    dx = x[1] - x[0]

    d0 = np.where(x < 0.0, ul[0], ur[0])
    u0 = np.where(x < 0.0, ul[1], ur[1])
    U = np.vstack((d0, u0))

    t = 0.0

    while t < t_final:
        dt = CFL * dx / max_speed(U[0])

        if t + dt > t_final:
            dt = t_final - t

        F = flux(U)

        UL = U[:, :-1]
        UR = U[:, 1:]
        FL = F[:, :-1]
        FR = F[:, 1:]

        # Lax-Friedrichs numerical flux
        flux_half = (dx / (2.0 * dt)) * (UL - UR) + 0.5 * (FL + FR)

        # Conservative update
        U[:, 1:-1] = U[:, 1:-1] - (dt / dx) * (flux_half[:, 1:] - flux_half[:, :-1])

        # Zero-gradient boundary conditions
        U[:, 0] = U[:, 1]
        U[:, -1] = U[:, -2]

        t += dt

    return x, U, dx



# Error norms

def compute_norms(error, dx):
    L1 = np.sum(np.abs(error)) * dx
    L2 = np.sqrt(np.sum(error**2) * dx)
    Linf = np.max(np.abs(error))
    return L1, L2, Linf



# Grid refinement study

grid_sizes = [125, 250, 500, 1000, 2000, 4000]

results = []

for N in grid_sizes:
    x, U_num, dx = solve_lf(N)
    U_ex = exact_solution(x)

    err_d = U_num[0] - U_ex[0]
    err_u = U_num[1] - U_ex[1]

    L1_d, L2_d, Linf_d = compute_norms(err_d, dx)
    L1_u, L2_u, Linf_u = compute_norms(err_u, dx)

    results.append({
        "N": N,
        "dx": dx,
        "L1_d": L1_d,
        "L2_d": L2_d,
        "Linf_d": Linf_d,
        "L1_u": L1_u,
        "L2_u": L2_u,
        "Linf_u": Linf_u
    })

# Convert to DataFrame
df = pd.DataFrame(results)

# orders of convergence

for col in ["L1_d", "L2_d", "Linf_d", "L1_u", "L2_u", "Linf_u"]:
    order_col = "order_" + col
    df[order_col] = np.nan
    for i in range(1, len(df)):
        E_coarse = df.loc[i - 1, col]
        E_fine = df.loc[i, col]
        dx_coarse = df.loc[i - 1, "dx"]
        dx_fine = df.loc[i, "dx"]
        df.loc[i, order_col] = np.log(E_coarse / E_fine) / np.log(dx_coarse / dx_fine)

# Print results
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df.round(6))


