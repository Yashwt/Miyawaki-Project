import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Input workbook must contain the recorded Miyawaki measurements.
DATA_FILE = "Recorded Data_Miyawaki Forest.xlsx"
N = 20000
rng = np.random.default_rng(20261102)
data = pd.read_excel(DATA_FILE)

# Course-guide uncertainty assumptions:
# H: normal, CV=10%; D: normal, CV=2.5%;
# species wood density: lognormal, CV=10%, shared within species;
# equation multiplier: lognormal, CV=1%, shared;
# equation residual: lognormal, sigma(log)=0.357, independent per stem.

sig_rho = np.sqrt(np.log(1 + 0.10**2))
rho_draws = {}
for sp in data["Tree Name"].unique():
    rho0 = float(data.loc[data["Tree Name"] == sp, "Wood_density (g/cm3)"].iloc[0])
    rho_draws[sp] = rho0*np.exp(rng.normal(-0.5*sig_rho**2, sig_rho, N))

sig_eq = np.sqrt(np.log(1 + 0.01**2))
eq_multiplier = np.exp(rng.normal(-0.5*sig_eq**2, sig_eq, N))
sigma_residual = 0.357

# Each tuple is (DBH cm, height m, number of identical culms/stems).
# Bamboo record biomass is calculated for one culm and multiplied by culm count.
# Multi-stem records 598 and 599 are summed stem by stem.
stems = {}
for _, r in data.iterrows():
    tid = int(r["Tree Id"])
    if tid == 578:
        stems[tid] = [(2.48, 5.41, 43)]
    elif tid == 584:
        stems[tid] = [(2.67, 4.74, 31)]
    elif tid == 598:
        stems[tid] = [(3.50, 3.19, 1), (3.18, 3.23, 1)]
    elif tid == 599:
        stems[tid] = [(3.47, 4.91, 1), (6.01, 5.01, 1)]
    elif pd.notna(r["DBH(in cm)"]):
        stems[tid] = [(float(r["DBH(in cm)"]), float(r["Height(in m)"]), 1)]

tree_draws = {}
for _, r in data.iterrows():
    tid = int(r["Tree Id"])
    if tid not in stems:
        continue
    total = np.zeros(N)
    rho = rho_draws[r["Tree Name"]]
    for D0, H0, count in stems[tid]:
        D = np.maximum(rng.normal(D0, D0*0.025, N), 1e-9)
        H = np.maximum(rng.normal(H0, H0*0.10, N), 1e-9)
        agb = 0.0673*(rho*D**2*H)**0.976
        residual = np.exp(rng.normal(-0.5*sigma_residual**2, sigma_residual, N))
        total += agb*eq_multiplier*residual*count
    tree_draws[tid] = total

rows = []
for tid, values in tree_draws.items():
    rows.append([tid, values.mean(), np.median(values),
                 np.quantile(values, .025), np.quantile(values, .975),
                 100*values.std()/values.mean()])
out = pd.DataFrame(rows, columns=["Tree ID", "Mean kg", "Median kg",
                                  "2.5 percentile kg", "97.5 percentile kg", "CV %"])
out.to_csv("Miyawaki_Part3_results.csv", index=False)

total = np.sum(np.vstack(list(tree_draws.values())), axis=0)
print("Total mean:", total.mean(), "kg")
print("Total 95% interval:", np.quantile(total, [.025, .975]), "kg")

# Plot (Updated for precise alignment and formatting)
out = out.sort_values("Tree ID")
tree_ids = out["Tree ID"].astype(str).tolist()
mean = out["Mean kg"].to_numpy()
lo = out["2.5 percentile kg"].to_numpy()
hi = out["97.5 percentile kg"].to_numpy()

x_pos = np.arange(len(tree_ids))

plt.figure(figsize=(12, 6))
plt.errorbar(x_pos, mean, yerr=[mean - lo, hi - mean], fmt="o", capsize=3, color="tab:blue")
plt.xticks(x_pos, tree_ids, rotation=0)
plt.xlabel("Tree ID")
plt.ylabel("Above-ground biomass (kg)")
plt.title("Monte Carlo biomass estimates (mean and 95% interval)")
plt.grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig("Miyawaki_Part3_Uncertainty_Plot.png", dpi=300)
plt.show()