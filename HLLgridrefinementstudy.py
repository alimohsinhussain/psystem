import numpy as np
import matplotlib.pyplot as plt


gamma = 2.0
CFL = 0.9
t_final = 1.0

ul = [1.0, 0.0]
ur = [0.7, 0.0]

# Exact solution constants
dL = 1.0
d_star = 0.830282
dR = 0.7

uL = 0.0
u_star = -0.276542
uR = 0.0

s1 = -1.629421
x_fan_left = 1.869290
x_fan_right = 2.414726

def flux(U):
    d = U[0]
    u = U[1]
    p = d**(-gamma)
    return np.vstack((-u, p))


def sound_speed(d):
    d_safe = np.maximum(d, 1.0e-12)
    return np.sqrt(gamma) * d_safe**(-(gamma + 1.0) / 2.0)


def max_speed(U):
    return np.max(sound_speed(U[0]))


def initial_condition(x):
    d = np.where(x < 0.0, ul[0], ur[0])
    u = np.where(x < 0.0, ul[1], ur[1])
    return np.vstack((d, u))

def exact_solution(x, t):
    xi = x / t

    d_exact = np.zeros_like(x)
    u_exact = np.zeros_like(x)

    #  left state
    mask1 = xi < s1
    d_exact[mask1] = dL
    u_exact[mask1] = uL

    # intermediate state
    mask2 = (xi >= s1) & (xi < x_fan_left)
    d_exact[mask2] = d_star
    u_exact[mask2] = u_star

    #  2-rarefaction fan
    mask3 = (xi >= x_fan_left) & (xi <= x_fan_right)

    d_exact[mask3] = (np.sqrt(2.0) / xi[mask3])**(2.0 / 3.0)

    d_fan = d_exact[mask3]
    u_exact[mask3] = 2.0 * np.sqrt(2.0) * (
        d_fan**(-0.5) - dR**(-0.5)
    )

    # right state
    mask4 = xi > x_fan_right
    d_exact[mask4] = dR
    u_exact[mask4] = uR

    return np.vstack((d_exact, u_exact))



# HLL flux

def hll_flux(UL, UR):
    dL_local = UL[0]
    dR_local = UR[0]

    cL = sound_speed(dL_local)
    cR = sound_speed(dR_local)

    SL = np.minimum(-cL, -cR)
    SR = np.maximum(cL, cR)

    FL = flux(UL)
    FR = flux(UR)

    FHLL = np.zeros_like(FL)

    # all waves move right
    maskL = SL >= 0.0
    FHLL[:, maskL] = FL[:, maskL]

    #Case 2: all waves move left
    maskR = SR <= 0.0
    FHLL[:, maskR] = FR[:, maskR]

    # star region crosses interface
    maskM = ~(maskL | maskR)
    FHLL[:, maskM] = (
        SR[maskM] * FL[:, maskM]
        - SL[maskM] * FR[:, maskM]
        + SL[maskM] * SR[maskM] * (UR[:, maskM] - UL[:, maskM])
    ) / (SR[maskM] - SL[maskM])

    return FHLL



# HLL solver

def solve_hll(N):
    x = np.linspace(-3.0, 4.0, N)
    dx = x[1] - x[0]

    U = initial_condition(x)

    t = 0.0

    while t < t_final:
        dt = CFL * dx / max_speed(U)

        if t + dt > t_final:
            dt = t_final - t

        UL = U[:, :-1]
        UR = U[:, 1:]

        F_half = hll_flux(UL, UR)

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
            raise ValueError("Nonphysical state: d became non-positive.")

        t += dt

    return x, dx, U



# Error norms

def compute_error_norms(U_num, U_exact, dx):
    error = U_num - U_exact

    L1 = dx * np.sum(np.abs(error), axis=1)

    L2 = np.sqrt(dx * np.sum(error**2, axis=1))

    Linf = np.max(np.abs(error), axis=1)

    return L1, L2, Linf



# Grid refinement study

N_values = [250, 500, 1000, 2000, 4000]

results = []

for N in N_values:
    x, dx, U_HLL = solve_hll(N)

    U_ex = exact_solution(x, t_final)

    L1, L2, Linf = compute_error_norms(U_HLL, U_ex, dx)

    results.append({
        "N": N,
        "dx": dx,
        "L1_d": L1[0],
        "L2_d": L2[0],
        "Linf_d": Linf[0],
        "L1_u": L1[1],
        "L2_u": L2[1],
        "Linf_u": Linf[1],
    })



# Compute observed orders

for i in range(len(results)):
    if i == 0:
        results[i]["p_L1_d"] = np.nan
        results[i]["p_L2_d"] = np.nan
        results[i]["p_Linf_d"] = np.nan
        results[i]["p_L1_u"] = np.nan
        results[i]["p_L2_u"] = np.nan
        results[i]["p_Linf_u"] = np.nan
    else:
        old = results[i - 1]
        new = results[i]

        ratio_dx = old["dx"] / new["dx"]

        results[i]["p_L1_d"] = np.log(old["L1_d"] / new["L1_d"]) / np.log(ratio_dx)
        results[i]["p_L2_d"] = np.log(old["L2_d"] / new["L2_d"]) / np.log(ratio_dx)
        results[i]["p_Linf_d"] = np.log(old["Linf_d"] / new["Linf_d"]) / np.log(ratio_dx)

        results[i]["p_L1_u"] = np.log(old["L1_u"] / new["L1_u"]) / np.log(ratio_dx)
        results[i]["p_L2_u"] = np.log(old["L2_u"] / new["L2_u"]) / np.log(ratio_dx)
        results[i]["p_Linf_u"] = np.log(old["Linf_u"] / new["Linf_u"]) / np.log(ratio_dx)



# Print table

print("\nGrid refinement study for HLL flux")
print("=" * 120)

header = (
    f"{'N':>8} {'dx':>12} "
    f"{'L1(d)':>12} {'p':>8} "
    f"{'L2(d)':>12} {'p':>8} "
    f"{'Linf(d)':>12} {'p':>8} "
    f"{'L1(u)':>12} {'p':>8} "
    f"{'L2(u)':>12} {'p':>8} "
    f"{'Linf(u)':>12} {'p':>8}"
)

print(header)
print("-" * 120)

for r in results:
    print(
        f"{r['N']:8d} {r['dx']:12.6e} "
        f"{r['L1_d']:12.6e} {r['p_L1_d']:8.3f} "
        f"{r['L2_d']:12.6e} {r['p_L2_d']:8.3f} "
        f"{r['Linf_d']:12.6e} {r['p_Linf_d']:8.3f} "
        f"{r['L1_u']:12.6e} {r['p_L1_u']:8.3f} "
        f"{r['L2_u']:12.6e} {r['p_L2_u']:8.3f} "
        f"{r['Linf_u']:12.6e} {r['p_Linf_u']:8.3f}"
    )



# convergence plots

dx_values = np.array([r["dx"] for r in results])

L1_d_values = np.array([r["L1_d"] for r in results])
L2_d_values = np.array([r["L2_d"] for r in results])
Linf_d_values = np.array([r["Linf_d"] for r in results])

L1_u_values = np.array([r["L1_u"] for r in results])
L2_u_values = np.array([r["L2_u"] for r in results])
Linf_u_values = np.array([r["Linf_u"] for r in results])

plt.figure(figsize=(8, 5))
plt.loglog(dx_values, L1_d_values, "o-", label=r"$L^1(d)$")
plt.loglog(dx_values, L2_d_values, "s-", label=r"$L^2(d)$")
plt.loglog(dx_values, Linf_d_values, "^-", label=r"$L^\infty(d)$")
plt.xlabel(r"Step size")
plt.ylabel("Error norm")
plt.title(r"$d$")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8, 5))
plt.loglog(dx_values, L1_u_values, "o-", label=r"$L^1(u)$")
plt.loglog(dx_values, L2_u_values, "s-", label=r"$L^2(u)$")
plt.loglog(dx_values, Linf_u_values, "^-", label=r"$L^\infty(u)$")
plt.xlabel(r"Step size")
plt.ylabel("Error norm")
plt.title(r"$u$")
plt.legend()
plt.tight_layout()

plt.show()