# Latent Regime Detection in Financial Time Series via Gibbs Sampling

A Bayesian Markov Switching Model (MSM) implementation using Gibbs Sampling and Forward-Filtering Backward-Sampling (FFBS) to detect latent Bull and Bear market regimes in financial time series.

This project analyzes weekly log returns of the S&P 500 and NIFTY 50 indices and probabilistically identifies hidden market states characterized by distinct volatility and return structures.

---

## Project Overview

Financial markets exhibit regime-dependent behavior:
- periods of stable growth with low volatility (Bull Markets)
- periods of turbulence and negative drift (Bear Markets)

Traditional linear models struggle to capture these structural transitions.

This project implements:
- A Two-State Hidden Markov Model
- Bayesian Gibbs Sampling
- FFBS (Forward-Filtering Backward-Sampling)
- Regime probability estimation
- Volatility regime detection

The model jointly estimates:
- hidden regimes
- transition probabilities
- regime-specific means and variances
- smoothed state probabilities over time

---

## Methodology

### Conditional Gaussian Returns

Weekly log returns are assumed to follow:

```math
r_t | S_t = j \sim \mathcal{N}(\mu_j, \sigma_j^2)
```

where:
- \(S_t\) is the latent market state
- \(j \in \{0,1\}\)

---

### Hidden Markov Dynamics

The latent states evolve according to a first-order Markov chain:

```math
P(S_t=j \mid S_{t-1}=i)=p_{ij}
```

---

### Bayesian Inference

Conjugate priors are used for:
- regime means
- variances
- transition probabilities

This enables exact Gibbs updates for all parameters.

---

## Core Techniques

### Forward-Filtering Backward-Sampling (FFBS)

Used to jointly sample the entire latent state sequence efficiently.

Advantages:
- avoids severe autocorrelation
- improves MCMC mixing
- enables scalable latent regime inference

---

### Gibbs Sampling

The Gibbs sampler iteratively samples:
1. Hidden states
2. Regime means
3. Regime variances
4. Transition probabilities

---

## Dataset

### S&P 500
- Weekly log returns
- 1110 observations

### NIFTY 50
- Weekly log returns
- 969 observations

### Additional Macro Context
- NBER US recession indicators used for economic comparison

---

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy

---

## Results

### S&P 500 Regimes

| Regime | Mean Return | Variance |
|---|---|---|
| Bull Market | 0.0031 | 0.000272 |
| Bear Market | -0.0049 | 0.002069 |

---

### NIFTY 50 Regimes

| Regime | Mean Return | Variance |
|---|---|---|
| Bull Market | 0.0024 | 0.000398 |
| Bear Market | -0.0025 | 0.002790 |

---

## Key Findings

### Accurate Crisis Detection
The model successfully identifies:
- 2008 Global Financial Crisis
- COVID-19 crash
- recession-linked volatility spikes

---

### Asymmetric Volatility
Bear regimes consistently exhibit:
- significantly higher variance
- negative or near-zero drift

---

### Cross-Market Contagion
NIFTY 50 demonstrates synchronized regime transitions with the S&P 500 during global crises, indicating strong international market coupling during downturns.

---

## Visualizations

The project generates:
- smoothed Bull/Bear regime probabilities
- index price overlays
- recession shading comparisons
- volatility regime transitions over time

---

## Repository Structure

```bash
├── data/
│   ├── sp500_data.csv
│   ├── nifty50_data.csv
│   └── us_recession.csv
│
├── notebooks/
│   └── regime_detection.ipynb
│
├── figures/
│   ├── sp500_regimes.png
│   └── nifty50_regimes.png
│
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Nagaveni5010/your-repository-name.git
cd your-repository-name
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch the notebook:

```bash
jupyter notebook
```

Run:
- `regime_detection.ipynb`

---

## Learning Outcomes

This project demonstrates:
- Bayesian inference for time series
- Hidden Markov Models (HMMs)
- Gibbs Sampling
- FFBS algorithms
- latent state estimation
- financial regime modelling
- volatility dynamics in equity markets

---

## Future Improvements

Potential extensions:
- multi-state regime models
- stochastic volatility integration
- particle filtering
- time-varying transition probabilities
- macroeconomic covariates
- sector-level regime decomposition

---

## Author

### Budati Nagaveni
B.Tech DSEB, Plaksha University

- LinkedIn: https://www.linkedin.com/in/budati-nagaveni-20b49235b/
- GitHub: https://github.com/Nagaveni5010

---
