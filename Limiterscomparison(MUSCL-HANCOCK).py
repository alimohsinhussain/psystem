import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ul = [1.0, 0.0]
ur = [0.7, 0.0]

gamma = 2.0
CFL = 0.5
d_floor = 1e-12
t_final = 1.0


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

    # intermediate state
    mask2 = (x >= s1) & (x < x_fan_left)
    d_exact[mask2] = d_star
    u_exact[mask2] = u_star

    # rarefaction fan
    mask3 = (x >= x_fan_left) & (x <= x_fan_right)
    d_exact[mask3] = (np.sqrt(2.0) / x[mask3]) ** (2.0 / 3.0)

    d_fan = d_exact[mask3]
    u_exact[mask3] = 2.0 * np.sqrt(2.0) * (d_fan ** (-0.5) - dR ** (-0.5))

    # right state
    mask4 = x > x_fan_right
    d_exact[mask4] = dR
    u_exact[mask4] = uR

    return np.vstack((d_exact, u_exact))


def flux(U):
    d = U[0]
    u = U[1]
    p = d ** (-gamma)
    return np.vstack((-u, p))


def sound_speed(d):
    return np.sqrt(gamma) * d ** (-(gamma + 1.0) / 2.0)


def max_speed(U):
    return np.max(sound_speed(U[0]))



# HLL flux

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



# Limiter functions

def limiter_phi(r, kind="minmod"):
    if kind == "minmod":
        return np.maximum(0.0, np.minimum(1.0, r))

    elif kind == "superbee":
        return np.maximum(
            0.0,
            np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0))
        )

    elif kind == "vanleer":
        return (r + np.abs(r)) / (1.0 + np.abs(r))

    else:
        raise ValueError(f"Unknown limiter: {kind}")


def compute_limited_slope(U, limiter="minmod", eps=1e-12):
    delta = np.zeros_like(U)

    dUm = U[:, 1:-1] - U[:, :-2]   # Delta_{i-1/2}
    dUp = U[:, 2:]   - U[:, 1:-1]  # Delta_{i+1/2}

    r = dUm / (dUp + eps)
    phi = limiter_phi(r, kind=limiter)

    delta[:, 1:-1] = phi * dUp
    return delta



# MUSCL-Hancock solver

def solve_muscl_hancock(N, limiter="minmod"):
    x = np.linspace(-3.0, 4.0, N)
    dx = x[1] - x[0]

    d = np.where(x < 0.0, ul[0], ur[0])
    u = np.where(x < 0.0, ul[1], ur[1])
    U = np.vstack((d, u))

    t = 0.0

    while t < t_final:
        dt = CFL * dx / max_speed(U)
        if t + dt > t_final:
            dt = t_final - t

        # 1. Limited slopes
        delta = compute_limited_slope(U, limiter=limiter)

        # 2. Piecewise linear reconstruction
        U_L = U - 0.5 * delta
        U_R = U + 0.5 * delta

        # positivity floor for d
        U_L[0] = np.maximum(U_L[0], d_floor)
        U_R[0] = np.maximum(U_R[0], d_floor)

        # 3. Hancock predictor
        F_L = flux(U_L)
        F_R = flux(U_R)

        U_L_bar = U_L + 0.5 * (dt / dx) * (F_L - F_R)
        U_R_bar = U_R + 0.5 * (dt / dx) * (F_L - F_R)

        U_L_bar[0] = np.maximum(U_L_bar[0], d_floor)
        U_R_bar[0] = np.maximum(U_R_bar[0], d_floor)

        # 4. Interface states
        UL_half = U_R_bar[:, :-1]
        UR_half = U_L_bar[:, 1:]

        # 5. HLL flux
        F_half = hll_flux(UL_half, UR_half)

        # 6. Conservative update
        U_new = U.copy()
        U_new[:, 1:-1] = U[:, 1:-1] - (dt / dx) * (F_half[:, 1:] - F_half[:, :-1])

        # zero gradient boundary conditions
        U_new[:, 0] = U_new[:, 1]
        U_new[:, -1] = U_new[:, -2]

        if np.any(U_new[0] <= 0):
            raise ValueError(f"Nonphysical state: d became non-positive with limiter {limiter}")

        U = U_new
        t += dt

    return x, U, dx

# Error norms

def compute_norms(error, dx):
    L1 = np.sum(np.abs(error)) * dx
    L2 = np.sqrt(np.sum(error ** 2) * dx)
    Linf = np.max(np.abs(error))
    return L1, L2, Linf



# Complete limiter comparison at one grid

limiters = ["minmod", "vanleer", "superbee"]
N_plot = 1000

solutions = {}
error_rows = []

for lim in limiters:
    x, U_num, dx = solve_muscl_hancock(N_plot, limiter=lim)
    U_ex = exact_solution(x)

    err_d = U_num[0] - U_ex[0]
    err_u = U_num[1] - U_ex[1]

    L1_d, L2_d, Linf_d = compute_norms(err_d, dx)
    L1_u, L2_u, Linf_u = compute_norms(err_u, dx)

    solutions[lim] = {
        "x": x,
        "U_num": U_num,
        "U_ex": U_ex,
        "dx": dx
    }

    error_rows.append({
        "Limiter": lim,
        "L1_d": L1_d,
        "L2_d": L2_d,
        "Linf_d": Linf_d,
        "L1_u": L1_u,
        "L2_u": L2_u,
        "Linf_u": Linf_u
    })

df_limiters = pd.DataFrame(error_rows)
print("\nLimiter comparison at N = 1000")
print(df_limiters.round(6))


# Grid refinement study for each limiter

grid_sizes = [125, 250, 500, 1000, 2000, 4000]
all_refinement = {}

for lim in limiters:
    rows = []

    for N in grid_sizes:
        x, U_num, dx = solve_muscl_hancock(N, limiter=lim)
        U_ex = exact_solution(x)

        err_d = U_num[0] - U_ex[0]
        err_u = U_num[1] - U_ex[1]

        L1_d, L2_d, Linf_d = compute_norms(err_d, dx)
        L1_u, L2_u, Linf_u = compute_norms(err_u, dx)

        rows.append({
            "N": N,
            "dx": dx,
            "L1_d": L1_d,
            "L2_d": L2_d,
            "Linf_d": Linf_d,
            "L1_u": L1_u,
            "L2_u": L2_u,
            "Linf_u": Linf_u
        })

    df = pd.DataFrame(rows)

    for col in ["L1_d", "L2_d", "Linf_d", "L1_u", "L2_u", "Linf_u"]:
        order_col = "order_" + col
        df[order_col] = np.nan
        for i in range(1, len(df)):
            E_coarse = df.loc[i - 1, col]
            E_fine = df.loc[i, col]
            dx_coarse = df.loc[i - 1, "dx"]
            dx_fine = df.loc[i, "dx"]

            df.loc[i, order_col] = np.log(E_coarse / E_fine) / np.log(dx_coarse / dx_fine)

    all_refinement[lim] = df

print("\nGrid refinement table for each limiter")
for lim in limiters:
    print("\n" + "=" * 80)
    print(f"Limiter: {lim}")
    print(all_refinement[lim].round(6))


# Choose exact solution from one stored result

x_ref = solutions["minmod"]["x"]
U_ex_ref = solutions["minmod"]["U_ex"]


# Full-domain comparison plots

plt.figure(figsize=(10, 4.5))
plt.plot(x_ref, U_ex_ref[0], 'k-', linewidth=2.5, label='Exact')
for lim in limiters:
    plt.plot(solutions[lim]["x"], solutions[lim]["U_num"][0], linewidth=1.6, label=lim)
plt.xlabel("x")
plt.ylabel("d(x,1)")
plt.title("Limiter comparison for d")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 4.5))
plt.plot(x_ref, U_ex_ref[1], 'k-', linewidth=2.5, label='Exact')
for lim in limiters:
    plt.plot(solutions[lim]["x"], solutions[lim]["U_num"][1], linewidth=1.6, label=lim)
plt.xlabel("x")
plt.ylabel("u(x,1)")
plt.title("Limiter comparison for u")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()



# Shock zoom

plt.figure(figsize=(10, 4.5))
plt.plot(x_ref, U_ex_ref[0], 'k-', linewidth=2.5, label='Exact')
for lim in limiters:
    plt.plot(solutions[lim]["x"], solutions[lim]["U_num"][0], linewidth=1.6, label=lim)
plt.xlim(-2.0, -1.1)
plt.xlabel("x")
plt.ylabel("d(x,1)")
plt.title("Shock-region zoom")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 4.5))
plt.plot(x_ref, U_ex_ref[1], 'k-', linewidth=2.5, label='Exact')
for lim in limiters:
    plt.plot(solutions[lim]["x"], solutions[lim]["U_num"][1], linewidth=1.6, label=lim)
plt.xlim(-2.0, -1.1)
plt.xlabel("x")
plt.ylabel("u(x,1)")
plt.title("Shock-region zoom")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()



# Rarefaction zoom

plt.figure(figsize=(10, 4.5))
plt.plot(x_ref, U_ex_ref[0], 'k-', linewidth=2.5, label='Exact')
for lim in limiters:
    plt.plot(solutions[lim]["x"], solutions[lim]["U_num"][0], linewidth=1.6, label=lim)
plt.xlim(1.7, 2.6)
plt.xlabel("x")
plt.ylabel("d(x,1)")
plt.title("Rarefaction-region zoom")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 4.5))
plt.plot(x_ref, U_ex_ref[1], 'k-', linewidth=2.5, label='Exact')
for lim in limiters:
    plt.plot(solutions[lim]["x"], solutions[lim]["U_num"][1], linewidth=1.6, label=lim)
plt.xlim(1.7, 2.6)
plt.xlabel("x")
plt.ylabel("u(x,1)")
plt.title("Rarefaction-region zoom")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()



# Error bar charts for L1

plt.figure(figsize=(8, 4.5))
plt.bar(df_limiters["Limiter"], df_limiters["L1_d"])
plt.ylabel("L1 error in d")
plt.ylim(0, 0.005)
plt.title("Limiter comparison: L1(d)")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

plt.figure(figsize=(8, 4.5))
plt.bar(df_limiters["Limiter"], df_limiters["L1_u"])
plt.ylabel("L1 error in u")
plt.ylim(0, 0.005)
plt.title("Limiter comparison: L1(u)")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()


# Error bar charts for L2

plt.figure(figsize=(8, 4.5))
plt.bar(df_limiters["Limiter"], df_limiters["L2_d"])
plt.ylabel("L2 error in d")
plt.ylim(0, 0.015)
plt.title("Limiter comparison: L2(d)")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

plt.figure(figsize=(8, 4.5))
plt.bar(df_limiters["Limiter"], df_limiters["L2_u"])
plt.ylabel("L2 error in u")
plt.ylim(0, 0.015)
plt.title("Limiter comparison: L2(u)")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()



# Log-log convergence plots for L1 


plt.figure(figsize=(9, 4.5))

for lim in limiters:
    plt.loglog(all_refinement[lim]["dx"], all_refinement[lim]["L1_d"], marker='o', label=lim)

# reference lines anchored at finest-grid error of minmod
dx_ref = all_refinement["minmod"]["dx"].values
L1_ref_d = all_refinement["minmod"]["L1_d"].values[-1]

ref_order1_d = L1_ref_d * (dx_ref / dx_ref[-1])**1.0
ref_orderhalf_d = L1_ref_d * (dx_ref / dx_ref[-1])**0.5

plt.loglog(dx_ref, ref_order1_d, 'k--', linewidth=1.5, label='Order 1')
plt.loglog(dx_ref, ref_orderhalf_d, 'k:', linewidth=1.5, label='Order 1/2')

plt.xlabel("dx")
plt.ylabel("L1 error in d")
plt.title("Grid refinement comparison: L1(d)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()


plt.figure(figsize=(9, 4.5))

for lim in limiters:
    plt.loglog(all_refinement[lim]["dx"], all_refinement[lim]["L1_u"], marker='o', label=lim)

# reference lines anchored at finest-grid error of minmod
dx_ref = all_refinement["minmod"]["dx"].values
L1_ref_u = all_refinement["minmod"]["L1_u"].values[-1]

ref_order1_u = L1_ref_u * (dx_ref / dx_ref[-1])**1.0
ref_orderhalf_u = L1_ref_u * (dx_ref / dx_ref[-1])**0.5

plt.loglog(dx_ref, ref_order1_u, 'k--', linewidth=1.5, label='Order 1')
plt.loglog(dx_ref, ref_orderhalf_u, 'k:', linewidth=1.5, label='Order 1/2')

plt.xlabel("dx")
plt.ylabel("L1 error in u")
plt.title("Grid refinement comparison: L1(u)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

# Log-log convergence plots for L2 with reference lines

plt.figure(figsize=(9, 4.5))

for lim in limiters:
    plt.loglog(all_refinement[lim]["dx"], all_refinement[lim]["L2_d"], marker='o', label=lim)

# reference lines anchored at finest-grid error of minmod
dx_ref = all_refinement["minmod"]["dx"].values
L2_ref_d = all_refinement["minmod"]["L2_d"].values[-1]

ref_order1_d = L2_ref_d * (dx_ref / dx_ref[-1])**1.0
ref_orderhalf_d = L2_ref_d * (dx_ref / dx_ref[-1])**0.5

plt.loglog(dx_ref, ref_order1_d, 'k--', linewidth=1.5, label='Order 1')
plt.loglog(dx_ref, ref_orderhalf_d, 'k:', linewidth=1.5, label='Order 1/2')

plt.xlabel("dx")
plt.ylabel("L2 error in d")
plt.title("Grid refinement comparison: L2(d)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()


plt.figure(figsize=(9, 4.5))

for lim in limiters:
    plt.loglog(all_refinement[lim]["dx"], all_refinement[lim]["L2_u"], marker='o', label=lim)

# reference lines anchored at finest-grid error of minmod
dx_ref = all_refinement["minmod"]["dx"].values
L2_ref_u = all_refinement["minmod"]["L2_u"].values[-1]

ref_order1_u = L2_ref_u * (dx_ref / dx_ref[-1])**1.0
ref_orderhalf_u = L2_ref_u * (dx_ref / dx_ref[-1])**0.5

plt.loglog(dx_ref, ref_order1_u, 'k--', linewidth=1.5, label='Order 1')
plt.loglog(dx_ref, ref_orderhalf_u, 'k:', linewidth=1.5, label='Order 1/2')

plt.xlabel("dx")
plt.ylabel("L2 error in u")
plt.title("Grid refinement comparison: L2(u)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()


# Compact summary table

summary_rows = []
for lim in limiters:
    df = all_refinement[lim]
    summary_rows.append({
        "Limiter": lim,
        "L1_d@N1000": df.loc[df["N"] == 1000, "L1_d"].values[0],
        "L1_u@N1000": df.loc[df["N"] == 1000, "L1_u"].values[0],
        "Final order L1_d": df["order_L1_d"].iloc[-1],
        "Final order L1_u": df["order_L1_u"].iloc[-1],
        "Linf_d@N1000": df.loc[df["N"] == 1000, "Linf_d"].values[0],
        "Linf_u@N1000": df.loc[df["N"] == 1000, "Linf_u"].values[0],
    })

df_summary = pd.DataFrame(summary_rows)
print("\nSummary table")
print(df_summary.round(6))

plt.show()