# Monte Carlo Option Pricing

Option pricing using Monte Carlo simulation, written in Python, with GPU acceleration via CUDA (using CuPy / Numba).

---

## Table of Contents

1. [What is an Option?](#what-is-an-option)
2. [Option Payoffs](#option-payoffs)
3. [The Black-Scholes Framework](#the-black-scholes-framework)
4. [Geometric Brownian Motion](#geometric-brownian-motion)
5. [Risk-Neutral Pricing](#risk-neutral-pricing)
6. [Monte Carlo Simulation](#monte-carlo-simulation)
7. [The Algorithm](#the-algorithm)
8. [Variance Reduction Techniques](#variance-reduction-techniques)
9. [CUDA Acceleration](#cuda-acceleration)
10. [References](#references)

---

## What is an Option?

A **financial option** is a derivative contract that gives the holder the *right, but not the obligation*, to buy or sell an underlying asset at a predetermined price (the **strike price** $K$) on or before a specified date (the **expiry** $T$).

| Type | Right granted | Exercised when |
|------|--------------|----------------|
| **Call** | Buy the asset at $K$ | Asset price $S_T > K$ |
| **Put**  | Sell the asset at $K$ | Asset price $S_T < K$ |

Options can also be classified by exercise style:

- **European** – can only be exercised at expiry $T$.
- **American** – can be exercised at any time up to $T$.
- **Exotic** – path-dependent or have other non-standard features (Asian, barrier, lookback, …).

---

## Option Payoffs

For a **European call** option the payoff at expiry is:

$$\Pi_{\text{call}} = \max(S_T - K,\; 0)$$

For a **European put** option:

$$\Pi_{\text{put}} = \max(K - S_T,\; 0)$$

where $S_T$ is the spot price of the underlying asset at maturity $T$.

---

## The Black-Scholes Framework

The industry-standard model for equity options assumes the asset price $S_t$ follows **Geometric Brownian Motion** (GBM) under the real-world probability measure $\mathbb{P}$:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

where:

| Symbol | Meaning |
|--------|---------|
| $\mu$ | Expected (drift) rate of return |
| $\sigma$ | Volatility (annualised standard deviation of returns) |
| $W_t$ | Standard Brownian motion (Wiener process) under $\mathbb{P}$ |

Key model assumptions:

- Continuous trading with no transaction costs or taxes.
- No arbitrage opportunities.
- The risk-free interest rate $r$ is constant.
- The asset pays no dividends (or a continuous dividend yield $q$ can be incorporated).
- Log-returns are normally distributed and i.i.d.

---

## Geometric Brownian Motion

Applying **Itô's Lemma** to $f(S_t) = \ln S_t$ transforms the SDE into a linear equation with an exact closed-form solution:

$$\ln S_T = \ln S_0 + \left(\mu - \frac{\sigma^2}{2}\right)T + \sigma W_T$$

Hence the asset price at time $T$ is:

$$\boxed{S_T = S_0 \exp\!\left[\left(\mu - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T}\, Z\right]}$$

where $Z \sim \mathcal{N}(0, 1)$ is a standard normal random variable, because $W_T \sim \mathcal{N}(0, T)$.

This single equation lets us **directly simulate** $S_T$ without stepping through intermediate time points (for plain European options).

---

## Risk-Neutral Pricing

The **fundamental theorem of asset pricing** states that the fair (no-arbitrage) price of a derivative is its *expected discounted payoff* under the **risk-neutral measure** $\mathbb{Q}$:

$$V_0 = e^{-rT}\, \mathbb{E}^{\mathbb{Q}}\!\left[\Pi(S_T)\right]$$

Under $\mathbb{Q}$ the drift $\mu$ is replaced by the risk-free rate $r$ (or $r - q$ when a continuous dividend yield $q$ is present):

$$S_T = S_0 \exp\!\left[\left(r - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T}\, Z\right], \quad Z \sim \mathcal{N}(0,1)$$

For a European call this expectation yields the famous **Black-Scholes formula**:

$$C_0 = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

$$d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)\,T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

where $N(\cdot)$ is the standard normal CDF. Monte Carlo reproduces this result numerically and generalises to cases where no closed form exists.

---

## Monte Carlo Simulation

### Core Idea

The **Monte Carlo method** approximates the expectation $\mathbb{E}^{\mathbb{Q}}[\Pi]$ by the **sample mean** over $N$ independent simulated payoffs:

$$\hat{V}_0 = e^{-rT} \cdot \frac{1}{N}\sum_{i=1}^{N} \Pi\!\left(S_T^{(i)}\right)$$

By the **Law of Large Numbers**, $\hat{V}_0 \to V_0$ as $N \to \infty$.

### Statistical Error

The estimator has a standard error that decreases as $1/\sqrt{N}$:

$$\text{SE} = \frac{\hat{\sigma}_{\Pi}}{\sqrt{N}}$$

where $\hat{\sigma}_{\Pi}$ is the sample standard deviation of the discounted payoffs. A 95 % confidence interval for the true price is approximately:

$$\hat{V}_0 \pm 1.96 \cdot \text{SE}$$

To halve the standard error you need to quadruple the number of simulations — this $\mathcal{O}(N^{-1/2})$ convergence rate is the main drawback of plain Monte Carlo.

---

## The Algorithm

```
Input : S0, K, r, σ, T, N
Output: estimated option price V̂, standard error SE

1.  for i = 1 to N do
2.      Draw  Z_i  ~  N(0,1)
3.      Compute  S_T^(i) = S0 * exp((r - σ²/2)*T + σ*sqrt(T)*Z_i)
4.      Compute payoff  Π_i = max(S_T^(i) - K, 0)   [call]
                     or Π_i = max(K - S_T^(i), 0)   [put]
5.  end for
6.  V̂  = exp(-r*T) * mean(Π_i)
7.  SE = exp(-r*T) * std(Π_i) / sqrt(N)
8.  return V̂, SE
```

For **path-dependent options** (Asian, barrier, etc.) step 3 is replaced by a full discretisation of the GBM path using $M$ time steps $\Delta t = T/M$:

$$S_{t + \Delta t} = S_t \exp\!\left[\left(r - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\, Z\right]$$

---

## Variance Reduction Techniques

Plain Monte Carlo can be slow to converge. Several techniques reduce $\hat{\sigma}_{\Pi}$ (and therefore SE) without increasing $N$:

### Antithetic Variates

For each random draw $Z_i$, also evaluate the payoff using $-Z_i$. The two payoffs are negatively correlated, so their average has lower variance:

$$\Pi_i^{\text{AV}} = \frac{\Pi(S_T(+Z_i)) + \Pi(S_T(-Z_i))}{2}$$

### Control Variates

Use a correlated quantity with a known expectation (e.g. the underlying asset price $S_T$) to adjust the estimator:

$$\hat{V}_0^{\text{CV}} = \hat{V}_0 - c\,\bigl(\hat{\mathbb{E}}[S_T] - S_0 e^{rT}\bigr)$$

where $c$ is chosen to minimise variance.

### Quasi-Monte Carlo (Low-Discrepancy Sequences)

Replace pseudo-random uniform samples with deterministic **quasi-random** sequences (Sobol, Halton) that cover the sample space more uniformly, improving convergence to $\mathcal{O}((\log N)^d / N)$.

---

## CUDA Acceleration

Generating millions of independent GBM paths is **embarrassingly parallel** — each path simulation is independent of every other. This makes the algorithm an ideal candidate for GPU execution:

- Each CUDA thread simulates one (or a small batch of) price path(s).
- Random number generation is handled by `cuRAND` (C++) or `CuPy` / `Numba`'s random generators (Python).
- The final reduction (mean over payoffs) is performed with an efficient parallel reduction kernel.

Typical speed-ups of **100×–1000×** over single-threaded CPU code are achievable for large $N$, making real-time Greeks computation and calibration tractable.

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy, 81(3), 637–654.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Hull, J. C. (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
- Jäckel, P. (2002). *Monte Carlo Methods in Finance*. Wiley.
- NVIDIA CUDA Toolkit Documentation – [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
