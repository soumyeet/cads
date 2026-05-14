"""Generate all report figures for gibbs_fixed."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, invgamma, beta as beta_dist, probplot
import os

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
os.makedirs("figures", exist_ok=True)

RCOLOR = "#e74c3c"; SPC = "#3498db"; NFC = "#2ecc71"

# ── Load ──────────────────────────────────────────────────────────────────────
sp500 = pd.read_csv("sp500_data.csv", index_col="Date")
nifty = pd.read_csv("nifty50_data.csv", index_col="Date")
rec   = pd.read_csv("us_recession.csv", index_col="observation_date")
sp500.index = pd.to_datetime(sp500.index, utc=True).tz_localize(None)
nifty.index  = pd.to_datetime(nifty.index,  utc=True).tz_localize(None)
rec.index   = pd.to_datetime(rec.index)
sp500.dropna(subset=["Log_Return"], inplace=True)
nifty.dropna(subset=["Log_Return"],  inplace=True)

def shade(ax, r):
    mask = r["USREC"] == 1; in_r = False; st = None
    for d, v in mask.items():
        if v and not in_r:  st, in_r = d, True
        elif not v and in_r: ax.axvspan(st, d, color=RCOLOR, alpha=0.15, lw=0); in_r = False
    if in_r: ax.axvspan(st, r.index[-1], color=RCOLOR, alpha=0.15, lw=0)

# ── FFBS + Gibbs (corrected) ──────────────────────────────────────────────────
def ffbs(y, mu, sig2, p_diag):
    T = len(y)
    P_mat = np.array([[p_diag[0], 1-p_diag[0]], [1-p_diag[1], p_diag[1]]])
    P_filt = np.zeros((T, 2)); P_filt[0] = [0.5, 0.5]
    for t in range(1, T):
        P_pred = P_mat.T @ P_filt[t-1]
        L = np.array([norm.pdf(y[t], mu[j], np.sqrt(sig2[j])) for j in range(2)])
        raw = P_pred * L; s = raw.sum()
        P_filt[t] = raw/s if s > 0 else [0.5, 0.5]
    S = np.zeros(T, dtype=int); S[-1] = np.random.choice(2, p=P_filt[-1])
    for t in range(T-2, -1, -1):
        pb = P_filt[t] * P_mat[:, S[t+1]]; s = pb.sum()
        S[t] = np.random.choice(2, p=pb/s if s > 0 else [0.5, 0.5])
    return S

def gibbs_msm(y, n_iter=3000, burn_in=1000, mu0=0., tau2=1., alpha0=2., beta0=0.01, a0=8., b0=2.):
    T = len(y); med = np.median(y)
    mu = np.array([y[y < med].mean(), y[y >= med].mean()])
    sig2 = np.array([y.var(), y.var()]); p_diag = np.array([0.95, 0.95])
    samples = {"mu": [], "sigma2": [], "p": [], "S": []}
    for k in range(n_iter):
        S = ffbs(y, mu, sig2, p_diag)    # THE FIX
        for j in range(2):
            idx = S==j; nj = idx.sum()
            if nj == 0: continue
            prec = 1./tau2 + nj/sig2[j]; mn = (mu0/tau2 + y[idx].sum()/sig2[j]) / prec
            mu[j] = np.random.normal(mn, 1./np.sqrt(prec))
        for j in range(2):
            idx = S==j; nj = idx.sum()
            if nj == 0: continue
            sig2[j] = invgamma.rvs(alpha0+nj/2., scale=beta0+0.5*((y[idx]-mu[j])**2).sum())
        for j in range(2):
            njj = ((S[:-1]==j)&(S[1:]==j)).sum(); nalt = ((S[:-1]==j)&(S[1:]!=j)).sum()
            p_diag[j] = beta_dist.rvs(a0+njj, b0+nalt)
        if k >= burn_in:
            samples["mu"].append(mu.copy()); samples["sigma2"].append(sig2.copy())
            samples["p"].append(p_diag.copy()); samples["S"].append(S.copy())
    return samples

np.random.seed(42)
res_sp  = gibbs_msm(sp500["Log_Return"].values)
np.random.seed(42)
res_nif = gibbs_msm(nifty["Log_Return"].values)

def align(res):
    mu_a = np.array(res["mu"]); s2_a = np.array(res["sigma2"]); S_a = np.array(res["S"])
    ms = s2_a.mean(axis=0)
    if ms[0] > ms[1]:  bull_i, bear_i = 1, 0
    else:              bull_i, bear_i = 0, 1; S_a = 1 - S_a
    return S_a.mean(axis=0), mu_a[:,bull_i], mu_a[:,bear_i], s2_a[:,bull_i], s2_a[:,bear_i]

Pb_sp,  mb_sp,  mBr_sp,  s2b_sp,  s2Br_sp  = align(res_sp)
Pb_nif, mb_nif, mBr_nif, s2b_nif, s2Br_nif = align(res_nif)

# ── Fig 1: Prices ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
axes[0].plot(sp500.index, sp500["Close"], color=SPC, lw=1.2); axes[0].set_title("S&P 500 Weekly Close Prices"); axes[0].set_ylabel("USD"); shade(axes[0], rec)
axes[1].plot(nifty.index, nifty["Close"], color=NFC, lw=1.2); axes[1].set_title("NIFTY 50 Weekly Close Prices"); axes[1].set_ylabel("INR"); shade(axes[1], rec)
plt.tight_layout(); plt.savefig("figures/fig_prices.png", bbox_inches="tight"); plt.close(); print("✓ fig_prices")

# ── Fig 2: Returns ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for ax, df, c, t in [(axes[0], sp500, SPC, "S&P 500"), (axes[1], nifty, NFC, "NIFTY 50")]:
    ax.plot(df.index, df["Log_Return"], color=c, lw=0.7, alpha=0.85)
    ax.axhline(0, color="k", lw=0.7, ls="--"); ax.set_title(f"{t} Weekly Log Returns"); ax.set_ylabel("Log Return")
    shade(ax, rec)
plt.tight_layout(); plt.savefig("figures/fig_returns.png", bbox_inches="tight"); plt.close(); print("✓ fig_returns")

# ── Fig 3: Distributions ──────────────────────────────────────────────────────
from scipy.stats import norm as norm_dist
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, col, name, c in [(axes[0], sp500["Log_Return"], "S&P 500", SPC), (axes[1], nifty["Log_Return"], "NIFTY 50", NFC)]:
    sns.histplot(col, bins=60, kde=True, ax=ax, color=c, stat="density", alpha=0.6, line_kws={"lw":2})
    xs = np.linspace(col.min(), col.max(), 300)
    ax.plot(xs, norm_dist.pdf(xs, col.mean(), col.std()), "r--", lw=1.5, label="Normal fit")
    ax.axvline(0, color="k", lw=0.8, ls="--"); ax.set_title(f"{name} Log Return Distribution"); ax.legend()
plt.tight_layout(); plt.savefig("figures/fig_distributions.png", bbox_inches="tight"); plt.close(); print("✓ fig_distributions")

# ── Fig 4: Q-Q ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col, name, c in [(axes[0], sp500["Log_Return"], "S&P 500", SPC), (axes[1], nifty["Log_Return"], "NIFTY 50", NFC)]:
    (osm, osr), (slope, intercept, _) = probplot(col)
    ax.scatter(osm, osr, s=8, alpha=0.5, c=c); ax.plot(osm, slope*np.array(osm)+intercept, "r-", lw=1.5)
    ax.set_title(f"{name} — Q-Q Plot"); ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
plt.tight_layout(); plt.savefig("figures/fig_qq.png", bbox_inches="tight"); plt.close(); print("✓ fig_qq")

# ── Fig 5: Rolling vol ────────────────────────────────────────────────────────
W = 21
sp500["RV"] = sp500["Log_Return"].rolling(W).std() * np.sqrt(52) * 100
nifty["RV"]  = nifty["Log_Return"].rolling(W).std()  * np.sqrt(52) * 100
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for ax, df, c, t in [(axes[0], sp500, SPC, "S&P 500"), (axes[1], nifty, NFC, "NIFTY 50")]:
    ax.plot(df.index, df["RV"], color=c, lw=1)
    ax.fill_between(df.index, 0, df["RV"], color=c, alpha=0.2)
    ax.set_title(f"{t} — 21-Week Rolling Volatility (Annualised %)"); ax.set_ylabel("Vol (%)")
    shade(ax, rec)
plt.tight_layout(); plt.savefig("figures/fig_rolling_vol.png", bbox_inches="tight"); plt.close(); print("✓ fig_rolling_vol")

# ── Fig 6: S&P 500 Regime ─────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(14, 5.5))
ax1.plot(sp500.index, sp500["Close"], color="black", lw=1.2, label="S&P 500")
ax1.set_ylabel("Price (USD)", fontsize=11)
ax2 = ax1.twinx()
ax2.fill_between(sp500.index, 0, Pb_sp,    color=NFC,    alpha=0.40, label="P(Bull)")
ax2.fill_between(sp500.index, 0, 1-Pb_sp,  color=RCOLOR, alpha=0.30, label="P(Bear)")
ax2.set_ylim(0, 1); ax2.set_ylabel("Regime Probability", fontsize=11)
shade(ax2, rec)
ax1.set_title("S&P 500 — Regime Probabilities (Corrected Gibbs Sampler)", fontsize=13)
h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc="upper left", fontsize=9)
plt.tight_layout(); plt.savefig("figures/fig_sp500_regime.png", bbox_inches="tight"); plt.close(); print("✓ fig_sp500_regime")

# ── Fig 7: NIFTY Regime ───────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(14, 5.5))
ax1.plot(nifty.index, nifty["Close"], color="black", lw=1.2, label="NIFTY 50")
ax1.set_ylabel("Price (INR)", fontsize=11)
ax2 = ax1.twinx()
ax2.fill_between(nifty.index, 0, Pb_nif,   color="steelblue", alpha=0.40, label="P(Bull)")
ax2.fill_between(nifty.index, 0, 1-Pb_nif, color=RCOLOR,      alpha=0.30, label="P(Bear)")
ax2.set_ylim(0, 1); ax2.set_ylabel("Regime Probability", fontsize=11)
shade(ax2, rec)
ax1.set_title("NIFTY 50 — Regime Probabilities (Corrected Gibbs Sampler)", fontsize=13)
h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc="upper left", fontsize=9)
plt.tight_layout(); plt.savefig("figures/fig_nifty_regime.png", bbox_inches="tight"); plt.close(); print("✓ fig_nifty_regime")

# ── Fig 8: Trace ──────────────────────────────────────────────────────────────
mu_arr   = np.array(res_sp["mu"])
sig2_arr = np.array(res_sp["sigma2"])
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
axes[0,0].plot(mu_arr[:,0],   lw=0.5, color=RCOLOR); axes[0,0].set_title("Trace: $\\mu_0$ (Bear)")
axes[0,1].plot(mu_arr[:,1],   lw=0.5, color=NFC);    axes[0,1].set_title("Trace: $\\mu_1$ (Bull)")
axes[1,0].plot(sig2_arr[:,0], lw=0.5, color=RCOLOR); axes[1,0].set_title("Trace: $\\sigma^2_0$ (Bear)"); axes[1,0].set_xlabel("Post burn-in iteration")
axes[1,1].plot(sig2_arr[:,1], lw=0.5, color=NFC);    axes[1,1].set_title("Trace: $\\sigma^2_1$ (Bull)"); axes[1,1].set_xlabel("Post burn-in iteration")
plt.suptitle("S&P 500 — MCMC Trace Plots (Corrected Sampler)", fontsize=13, y=1.01)
plt.tight_layout(); plt.savefig("figures/fig_trace.png", bbox_inches="tight"); plt.close(); print("✓ fig_trace")

# ── Fig 9: Posteriors ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
for ax, s, t, c in [(axes[0,0], mu_arr[:,0],   "$\\mu_0$ (Bear)",       RCOLOR),
                    (axes[0,1], mu_arr[:,1],   "$\\mu_1$ (Bull)",       NFC),
                    (axes[1,0], sig2_arr[:,0], "$\\sigma^2_0$ (Bear)",  RCOLOR),
                    (axes[1,1], sig2_arr[:,1], "$\\sigma^2_1$ (Bull)",  NFC)]:
    ax.hist(s, bins=40, color=c, alpha=0.7, density=True, edgecolor="white")
    ax.axvline(s.mean(), color="k", lw=1.2, ls="--", label=f"mean={s.mean():.5f}")
    ax.set_title(f"Posterior: {t}"); ax.legend(fontsize=8)
plt.suptitle("S&P 500 — Posterior Distributions (Corrected Sampler)", fontsize=13, y=1.01)
plt.tight_layout(); plt.savefig("figures/fig_posteriors.png", bbox_inches="tight"); plt.close(); print("✓ fig_posteriors")

# ── Fig 10: Scatter ───────────────────────────────────────────────────────────
# Resample both to week-end Friday so dates align regardless of exchange calendars
sp_weekly  = sp500["Log_Return"].resample("W-FRI").last().dropna()
nif_weekly = nifty["Log_Return"].resample("W-FRI").last().dropna()

combined = pd.DataFrame({"S&P 500": sp_weekly, "NIFTY 50": nif_weekly}).dropna()
print(f"Scatter: {len(combined)} overlapping weeks")

r = combined["S&P 500"].corr(combined["NIFTY 50"])
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: scatter
axes[0].scatter(combined["S&P 500"], combined["NIFTY 50"],
                alpha=0.4, s=15, color="mediumpurple")
axes[0].axhline(0, color="k", lw=0.7); axes[0].axvline(0, color="k", lw=0.7)
axes[0].set_xlabel("S&P 500 Log Return"); axes[0].set_ylabel("NIFTY 50 Log Return")
axes[0].set_title(f"S&P 500 vs NIFTY 50  (r = {r:.3f})", fontsize=11)

# Right: 52-week rolling correlation
rolling_r = combined["S&P 500"].rolling(52).corr(combined["NIFTY 50"])
axes[1].plot(rolling_r.index, rolling_r, color="mediumpurple", lw=1.3)
axes[1].axhline(rolling_r.mean(), color="k", lw=0.9, ls="--",
                label=f"Mean r = {rolling_r.mean():.3f}")
axes[1].set_title("52-Week Rolling Correlation: S&P 500 & NIFTY 50", fontsize=11)
axes[1].set_ylabel("Pearson r"); axes[1].legend()

plt.tight_layout()
plt.savefig("figures/fig_scatter.png", bbox_inches="tight"); plt.close()
print(f"✓ fig_scatter  (r = {r:.4f})")

print(f"\nAll figures saved.")

