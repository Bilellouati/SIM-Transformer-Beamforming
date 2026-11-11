"""
Build the bit-depth comparison table and plot for the K=4 joint phase+power
model (1/2/3/4 bits), mirroring the K=1 Comparison_Report_BitQuantization
convention, with bit=2 highlighted as the project's standard/focus.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

results = {}
for b in [1, 2, 3, 4]:
    d = loadmat(f"transformer_K4_powerctrl_bit{b}_results.mat")
    results[b] = {
        "sum_rate": d["sum_rate"].flatten(),
        "per_user_rate": d["per_user_rate"],
    }

bits = [1, 2, 3, 4]
mean_sumrate = [results[b]["sum_rate"].mean() for b in bits]
mean_peruser = [results[b]["per_user_rate"].mean(axis=0) for b in bits]
fairness_gap = [results[b]["per_user_rate"].mean(axis=0).max() - results[b]["per_user_rate"].mean(axis=0).min() for b in bits]

print("Bit | Mean sum rate | Per-user mean rates | Fairness gap (max-min)")
for i, b in enumerate(bits):
    print(f"{b}   | {mean_sumrate[i]:.4f}        | {np.round(mean_peruser[i], 3)} | {fairness_gap[i]:.4f}")

plt.figure(figsize=(7, 5))
plt.plot(bits, mean_sumrate, "o-", color="steelblue", linewidth=2, markersize=8, label="Sum rate")
plt.scatter([2], [mean_sumrate[1]], color="crimson", s=150, zorder=5, label="Our focus (2-bit)")
plt.xlabel("Quantization resolution [bits]")
plt.ylabel("Mean sum rate [bit/s/Hz]")
plt.title("K=4 multi-user: sum rate vs. phase-shifter resolution\n(quantization-aware training + fairness + joint power control)")
plt.xticks(bits)
plt.legend()
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.savefig("comparison_bits_K4.png", dpi=150)

plt.figure(figsize=(7, 5))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for k in range(4):
    plt.plot(bits, [mean_peruser[i][k] for i in range(4)], "o-", color=colors[k], label=f"User {k+1}")
plt.scatter([2]*4, [mean_peruser[1][k] for k in range(4)], color="black", s=60, zorder=5, marker="x")
plt.xlabel("Quantization resolution [bits]")
plt.ylabel("Mean per-user rate [bit/s/Hz]")
plt.title("K=4 multi-user: per-user rate vs. resolution (fairness check)\nblack x marks = our focus (2-bit)")
plt.xticks(bits)
plt.legend()
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.savefig("comparison_peruser_bits_K4.png", dpi=150)

savemat("comparison_bits_K4_summary.mat", {
    "bits": bits, "mean_sum_rate": mean_sumrate,
    "mean_per_user_rate": np.array(mean_peruser), "fairness_gap": fairness_gap,
})
print("\nSaved: comparison_bits_K4.png, comparison_peruser_bits_K4.png, comparison_bits_K4_summary.mat")
