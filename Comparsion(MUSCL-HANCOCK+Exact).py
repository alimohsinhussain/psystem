import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ul = [1.0, 0.0]
ur = [0.7, 0.0]
CFL = 0.7
gamma = 2.0
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

    # Region 1: left state
    mask1 = x < s1
    d_exact[mask1] = dL
    u_exact[mask1] = uL

    # Region 2: intermediate state
    mask2 = (x >= s1) & (x < x_fan_left)
    d_exact[mask2] = d_star
    u_exact[mask2] = u_star

    # Region 3: rarefaction fan
    mask3 = (x >= x_fan_left) & (x <= x_fan_right)
    d_exact[mask3] = (np.sqrt(2.0) / x[mask3]) ** (2.0 / 3.0)

    d_fan = d_exact[mask3]
    u_exact[mask3] = 2.0 * np.sqrt(2.0) * (d_fan ** (-0.5) - dR ** (-0.5))

    # Region 4: right state
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


def compute_norms(error, dx):
    L1 = np.sum(np.abs(error)) * dx
    L2 = np.sqrt(np.sum(error ** 2) * dx)
    Linf = np.max(np.abs(error))
    return L1, L2, Linf

grid_sizes = [125, 250, 500, 1000, 2000, 4000]

results = []

for N in grid_sizes:
    x, U_num, dx = solve_muscl_hancock(N)
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

df = pd.DataFrame(results)


for col in ["L1_d", "L2_d", "Linf_d", "L1_u", "L2_u", "Linf_u"]:
    order_col = "order_" + col
    df[order_col] = np.nan
    for i in range(1, len(df)):
        E_coarse = df.loc[i - 1, col]
        E_fine = df.loc[i, col]
        dx_coarse = df.loc[i - 1, "dx"]
        dx_fine = df.loc[i, "dx"]

        df.loc[i, order_col] = np.log(E_coarse / E_fine) / np.log(dx_coarse / dx_fine)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
print(df.round(6))

x, U_num, dx = solve_muscl_hancock(1000)
U_ex = exact_solution(x)

plt.figure(figsize=(9, 4))
plt.plot(x, U_ex[0], label="Exact", linewidth=2)
plt.plot(x, U_num[0], label="MUSCL-Hancock + HLL")
plt.xlabel("x")
plt.ylabel("d(x,1)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(9, 4))
plt.plot(x, U_ex[1], label="Exact", linewidth=2)
plt.plot(x, U_num[1], label="MUSCL-Hancock + HLL")
plt.xlabel("x")
plt.ylabel("u(x,1)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


# Error study
# Data for d
N = np.array([125, 250, 500, 1000, 2000, 4000])
dx = np.array([0.056452, 0.028112, 0.014028, 0.007007, 0.003502, 0.00175])
L1_d = np.array([0.015743, 0.007815, 0.004059, 0.002118, 0.001, 0.000505])
L2_d = np.array([0.020747, 0.014196, 0.010081, 0.008035, 0.005218, 0.003438])
Linf_d = np.array([0.060701, 0.055368, 0.062777, 0.082135, 0.072422, 0.061783])

# Plot (log-log)
plt.figure()
plt.loglog(dx, L1_d, marker='o', label='L1 norm')
plt.loglog(dx, L2_d, marker='s', label='L2 norm')
plt.loglog(dx, Linf_d, marker='^', label='Linf norm')

plt.xlabel('Step size')
plt.ylabel('Error')
plt.title('d')
plt.legend()

# Data for u
N = np.array([125, 250, 500, 1000, 2000, 4000])
dx = np.array([0.056452, 0.028112, 0.014028, 0.007007, 0.003502, 0.00175])
L1_u = np.array([0.029321, 0.014282, 0.007392, 0.003988, 0.001806, 0.000929])
L2_u = np.array([0.036289, 0.023983, 0.016467, 0.013834, 0.008268, 0.005745])
Linf_u = np.array([0.106383, 0.097885, 0.094148, 0.142222, 0.109572, 0.108142])

# Plot (log-log)
plt.figure()
plt.loglog(dx, L1_u, marker='o', label='L1 norm')
plt.loglog(dx, L2_u, marker='s', label='L2 norm')
plt.loglog(dx, Linf_u, marker='^', label='Linf norm')

plt.xlabel('dx')
plt.ylabel('Error')
plt.title('u')
plt.legend()
plt.show()