"""
PLVOM: simulation and estimation implementation.

Implementation of the Poisson Latent Variable Ordinal Model (PLVOM) for
multiclass link prediction in dynamic directed weighted networks.

Date: 2026-08-27
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


# Unpack the parameter vector
def unpack_params(params, n, p):
    S = params[:n]  # S1...Sn
    R_free = params[n:2 * n - 1]  # R1...Rn-1
    rho = params[2 * n - 1]  # ρ
    beta = params[2 * n:]  # β

    R = np.zeros(n)
    R[:n - 1] = R_free
    R[n - 1] = 0.0  # R_n = 0

    return S, R, rho, beta


# Convert arrays to dyad form, retaining only i != j
def prepare_dyad_data(Y, X):
    T, n, n2 = Y.shape  # Shape of Y: (T, n, n)
    mask = ~np.eye(n, dtype=bool)  # Boolean mask excluding diagonal entries (self-loops)
    senders, receivers = np.where(mask)  # Sender and receiver indices for directed off-diagonal dyads
    Y_flat = Y[:, mask]  # Implementation note.
    X_flat = X[:, mask, :]  # Implementation note.

    return Y_flat, X_flat, senders, receivers


# Compute ordinal category probabilities
def compute_category_probs(lam, alphas):
    m = lam.shape[0]  # Implementation note.
    K = len(alphas) - 1
    probs = np.zeros((m, K), dtype=float)  # Implementation note.

    # Probabilities for the first K-1 categories
    for k in range(K - 1):
        lower = alphas[k]
        upper = alphas[k + 1]
        probs[:, k] = poisson.cdf(upper, lam) - poisson.cdf(lower, lam)

    # Last category
    lower_last = alphas[K - 1]
    probs[:, K - 1] = 1.0 - poisson.cdf(lower_last, lam)

    return probs


# Negative log-likelihood
def neg_log_likelihood(params, n, Y_flat, X_flat, senders, receivers, alphas):
    T, m = Y_flat.shape  # Number of time points and dyads
    p = X_flat.shape[2]  # Number of covariates
    S, R, rho, beta = unpack_params(params, n, p)  # Unpack parameters

    nll = 0.0

    # Initialize the lagged response with Y(0) = 0
    lag_y = np.zeros(m, dtype=float)

    for t in range(T):
        # Observed categories for all dyads at time t
        y_t = Y_flat[t].astype(int)

        # Linear predictor
        eta = (
                S[senders]
                + R[receivers]
                + rho * np.log(lag_y + 1.0)
                + X_flat[t] @ beta
        )

        # Poisson intensity
        lam = np.exp(eta)

        # Category probabilities (m x K)
        probs = compute_category_probs(lam, alphas)

        # Probability of each observed category
        chosen_probs = probs[np.arange(m), y_t]

  # Implementation note.
        nll -= np.sum(np.log(chosen_probs))

        # Update the lagged response
        lag_y = y_t.astype(float)

    return nll


  # Implementation note.
# Split the node-effect and fixed-dimensional parameter blocks
def split_theta_psi(params, n, p):
    S, R, rho, beta = unpack_params(params, n, p)
    theta = np.concatenate([S, R[:n - 1]])
    psi = np.concatenate([[rho], beta])
    return theta, psi


# Recombine the node-effect and fixed-dimensional parameter blocks
def combine_theta_psi(theta, psi, n, p):
    full = np.zeros(2 * n + p, dtype=float)

  # Implementation note.
    full[:n] = theta[:n]
    full[n:2 * n - 1] = theta[n:2 * n - 1]

    # psi = [rho, beta]
    full[2 * n - 1] = psi[0]
    full[2 * n:] = psi[1:]

    return full


# Evaluate the Poisson PMF at a category boundary
def poisson_pmf_boundary(a, lam):
  # Implementation note.
    if np.isinf(a) or a < 0:
        return np.zeros_like(lam, dtype=float)
    return poisson.pmf(int(a), lam)


def compute_score_info(lam, y_t, alphas):
    """
    lam   : shape=(m,)
    y_t   : shape=(m,), observed categories
    alphas: [-1, a1, ..., inf]
    """
    probs = compute_category_probs(lam, alphas)  # Implementation note.
    m, K = probs.shape

  # Implementation note.
    hdot = np.zeros((m, K), dtype=float)

    for k in range(K):
        lower = alphas[k]
        upper = alphas[k + 1]

        p_lower = poisson_pmf_boundary(lower, lam)
        p_upper = poisson_pmf_boundary(upper, lam)

        # h'_k(eta) = lam * (p_alpha_k - p_alpha_{k+1})
  # Implementation note.
        hdot[:, k] = lam * (p_lower - p_upper)

    chosen_probs = probs[np.arange(m), y_t]
    chosen_hdot = hdot[np.arange(m), y_t]

    score_eta = chosen_hdot / chosen_probs  # Implementation note.
    info_eta = np.sum((hdot ** 2) / probs, axis=1)  # Implementation note.

    return score_eta, info_eta, probs


# With psi fixed, compute the score and Fisher information for theta
def theta_score_and_fisher(theta, psi, n, Y_flat, X_flat, senders, receivers, alphas):
    """
    theta = [S, R_free], 长度 2n-1
    psi   = [rho, beta], 长度 p+1
    """
    T, m = Y_flat.shape
    p = X_flat.shape[2]

    S = theta[:n]
    R_free = theta[n:]
    R = np.zeros(n, dtype=float)
    R[:n - 1] = R_free
    R[n - 1] = 0.0

    rho = psi[0]
    beta = psi[1:]

    score_S = np.zeros(n, dtype=float)
    score_R = np.zeros(n - 1, dtype=float)

  # Implementation note.
    W = np.zeros((n, n), dtype=float)

    lag_y = np.zeros(m, dtype=float)  # Implementation note.

    for t in range(T):
        y_t = Y_flat[t].astype(int)  # Implementation note.

        eta = (
                S[senders]
                + R[receivers]
                + rho * np.log(lag_y + 1.0)
                + X_flat[t] @ beta
        )
        lam = np.exp(eta)

  # Implementation note.
        score_eta, info_eta, probs = compute_score_info(lam, y_t, alphas)

  # Implementation note.
        np.add.at(score_S, senders, score_eta)

        mask_r = receivers < (n - 1)  # Implementation note.
        np.add.at(score_R, receivers[mask_r], score_eta[mask_r])

  # Implementation note.
        np.add.at(W, (senders, receivers), info_eta)

        lag_y = y_t.astype(float)

    # Construct the Fisher information matrix V
    d = 2 * n - 1
    V = np.zeros((d, d), dtype=float)

    # Upper-left block: diagonal entries for S
    for i in range(n):
        V[i, i] = np.sum(W[i, :]) - W[i, i]

    # Lower-right block: diagonal entries for R
    for j in range(n - 1):
        V[n + j, n + j] = np.sum(W[:, j]) - W[j, j]

    # Cross block
    for i in range(n):
        for j in range(n - 1):
            if i != j:
                V[i, n + j] = W[i, j]
                V[n + j, i] = W[i, j]

    U_theta = np.concatenate([score_S, score_R])

    return U_theta, V


# Construct the approximate inverse G for the class-L information matrix
def build_G_from_V(V, n):
    d = 2 * n - 1  # V: shape=(2n-1, 2n-1)

    v2n = np.zeros(d, dtype=float)  # Implementation note.

  # Implementation note.
    for a in range(d):
        v2n[a] = V[a, a] - (np.sum(V[a, :]) - V[a, a])

  # Implementation note.
    v2n2n = np.sum(v2n)

    # Allocate the approximate inverse matrix G
    G = np.zeros_like(V)

    # Upper-left block
    for a in range(n):
        for b in range(n):
            G[a, b] = (1.0 / v2n2n)
            if a == b:
                G[a, b] += 1.0 / V[a, a]

    # Lower-right block
    for a in range(n, d):
        for b in range(n, d):
            G[a, b] = (1.0 / v2n2n)
            if a == b:
                G[a, b] += 1.0 / V[a, a]

    # Cross block
    G[:n, n:] = -1.0 / v2n2n
    G[n:, :n] = -1.0 / v2n2n

    return G


# Optimize the theta block
def optimize_theta_block(
    theta_init,
    psi_fixed,
    n,
    Y_flat,
    X_flat,
    senders,
    receivers,
    alphas,
    maxiter=500,
    tol=1e-5,
    step_init=1.0
):
    theta = theta_init.copy()

    p = X_flat.shape[2]
    params_old = combine_theta_psi(theta, psi_fixed, n, p)
    nll_old = neg_log_likelihood(
        params_old, n, Y_flat, X_flat, senders, receivers, alphas
    )

    success = False
    message = "theta block reached maxiter"
    diff = np.nan

    for it in range(maxiter):
        U_theta, V = theta_score_and_fisher(
            theta, psi_fixed, n, Y_flat, X_flat, senders, receivers, alphas
        )

        G = build_G_from_V(V, n)
        delta = G @ U_theta

        step = step_init
        improved = False

        while step > 1e-8:
            theta_new = theta + step * delta

            params_new = combine_theta_psi(theta_new, psi_fixed, n, p)
            nll_new = neg_log_likelihood(
                params_new, n, Y_flat, X_flat, senders, receivers, alphas
            )

            if np.isfinite(nll_new) and (nll_new < nll_old):
                improved = True
                break

            step *= 0.5

        if not improved:
            message = "theta block stopped without improvement"
            break

        diff = np.max(np.abs(theta_new - theta))
        theta = theta_new
        nll_old = nll_new

        if diff < tol:
            success = True
            message = "theta block converged"
            break

    return {
        "x": theta,
        "success": success,
        "message": message,
        "n_iter": it + 1,
        "final_nll": nll_old,
        "final_diff": diff
    }


  # Implementation note.
def nll_and_grad_psi(psi, theta_fixed, n, Y_flat, X_flat, senders, receivers, alphas):
    T, m = Y_flat.shape
    p = X_flat.shape[2]

    S = theta_fixed[:n]
    R_free = theta_fixed[n:]
    R = np.zeros(n, dtype=float)
    R[:n - 1] = R_free
    R[n - 1] = 0.0

    rho = psi[0]
    beta = psi[1:]

    nll = 0.0
    grad_rho = 0.0
    grad_beta = np.zeros(p, dtype=float)

    lag_y = np.zeros(m, dtype=float)

    for t in range(T):
        y_t = Y_flat[t].astype(int)

        lag_term = np.log(lag_y + 1.0)
        eta = (
                S[senders]
                + R[receivers]
                + rho * lag_term
                + X_flat[t] @ beta
        )
        lam = np.exp(eta)

        score_eta, info_eta, probs = compute_score_info(lam, y_t, alphas)

        chosen_probs = probs[np.arange(m), y_t]
        nll -= np.sum(np.log(chosen_probs))

        # The gradient of the negative log-likelihood is minus the score
        grad_rho -= np.sum(score_eta * lag_term)
        grad_beta -= X_flat[t].T @ score_eta

        lag_y = y_t.astype(float)

    grad_psi = np.concatenate([[grad_rho], grad_beta])
    return nll, grad_psi


# Optimize the psi block
def optimize_psi_block(
        psi_init,
        theta_fixed,
        n,
        Y_flat,
        X_flat,
        senders,
        receivers,
        alphas,
        maxiter=500,
):
    p = X_flat.shape[2]

    def fun(ps):
        nll, grad = nll_and_grad_psi(
            ps, theta_fixed, n, Y_flat, X_flat, senders, receivers, alphas
        )
        return nll

    def jac(ps):
        nll, grad = nll_and_grad_psi(
            ps, theta_fixed, n, Y_flat, X_flat, senders, receivers, alphas
        )
        return grad

    result = minimize(
        fun=fun,
        x0=psi_init,
        method="L-BFGS-B",
        jac=jac,
        options={"maxiter": maxiter}
    )
    return result


def fit_dynamic_ordinal_poisson_two_step(
        Y,
        X,
        alphas,
        init_params,
        outer_maxiter=200,
        theta_maxiter=200,
        psi_maxiter=200,
        outer_tol=1e-5,
):
    T, n, _ = Y.shape

    p = X.shape[3]

    # Convert the data to off-diagonal dyad form
    Y_flat, X_flat, senders, receivers = prepare_dyad_data(Y, X)

    theta, psi = split_theta_psi(init_params, n, p)

    # Initialize the optimizer result holder
    psi_result = None

    # Outer loop alternating between the two parameter blocks
    for outer in range(outer_maxiter):
        theta_old = theta.copy()
        psi_old = psi.copy()

        # Step 1: update theta using the approximate inverse G
        theta_result = optimize_theta_block(
            theta_init=theta,
            psi_fixed=psi,
            n=n,
            Y_flat=Y_flat,
            X_flat=X_flat,
            senders=senders,
            receivers=receivers,
            alphas=alphas,
            maxiter=theta_maxiter
        )
        theta = theta_result["x"]

        # Step 2: update psi using L-BFGS-B
        psi_result = optimize_psi_block(
            psi_init=psi,
            theta_fixed=theta,
            n=n,
            Y_flat=Y_flat,
            X_flat=X_flat,
            senders=senders,
            receivers=receivers,
            alphas=alphas,
            maxiter=psi_maxiter,
        )
        psi = psi_result.x

        # Compute the outer-loop convergence criterion
        diff_theta = np.max(np.abs(theta - theta_old))
        diff_psi = np.max(np.abs(psi - psi_old))
        diff = max(diff_theta, diff_psi)

        if diff < outer_tol:
            break

    final_params = combine_theta_psi(theta, psi, n, p)
    S_hat, R_hat, rho_hat, beta_hat = unpack_params(final_params, n, p)

    return {
        "final_params": final_params,
        "S_hat": S_hat,
        "R_hat": R_hat,
        "rho_hat": rho_hat,
        "beta_hat": beta_hat,
        "theta_result": theta_result,
        "psi_result": psi_result,
        "Y_flat": Y_flat,
        "X_flat": X_flat,
        "senders": senders,
        "receivers": receivers
    }


# Simulate ordinal network data with temporal dependence under the model
def simulate_dynamic_ordinal_poisson_data(
        n,
        T,
        p,
        alphas,
        seed
):
    K = len(alphas) - 1
    rng = np.random.default_rng(seed)

    # Equally spaced sender effects over [-0.5, 0.5]
    S_true = np.linspace(-0.5, 0.5, n)
    R_true = np.zeros(n)
    R_true[:n - 1] = -0.5 + np.arange(n - 1) / (n - 1)
    R_true[-1] = 0.0

    rho_true = 0.5
    beta_true = np.array([0.5, -0.3, 0.2], dtype=float)

    # Covariate distribution
    X = rng.uniform(-1, 1, size=(T, n, n, p))
    #X = rng.normal(loc=0.0, scale=1.0, size=(T, n, n, p))
    #X = 0.5 * rng.laplace(loc=0.0, scale=1.0, size=(T, n, n, p))

    #X = np.zeros((T, n, n, p), dtype=float)

    # Covariate 1: Uniform(-1, 1)
    #X[..., 0] = rng.uniform(-1.0, 1.0, size=(T, n, n))

    # Covariate 2: standard normal N(0, 1)
    #X[..., 1] = rng.normal(loc=0.0, scale=1.0, size=(T, n, n))

    # Covariate 3: scaled Laplace distribution
    #X[..., 2] = 0.5 * rng.laplace(loc=0.0, scale=1.0, size=(T, n, n))

    Y = np.zeros((T, n, n), dtype=int)
    Z = np.zeros((T, n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            for t in range(T):
                lag_y = 0 if t == 0 else Y[t - 1, i, j]

                eta = (
                        S_true[i]
                        + R_true[j]
                        + rho_true * np.log(lag_y + 1.0)
                        + np.dot(X[t, i, j], beta_true)
                )

                lam = np.exp(eta)
                z = rng.poisson(lam)
                Z[t, i, j] = z

                y_val = None
                for k in range(K - 1):
                    if alphas[k] < z <= alphas[k + 1]:  # First K-1 categories
                        y_val = k
                        break
                if y_val is None:  # Last category
                    y_val = K - 1

                Y[t, i, j] = y_val

    return Y, Z, X, {
        "S_true": S_true,
        "R_true": R_true,
        "rho_true": rho_true,
        "beta_true": beta_true
    }


# Monte Carlo simulation
def monte_carlo_dynamic_ordinal_poisson(
        n,
        T,
        p,
        alphas,
        n_rep,
        seed_base=2026,
        outer_maxiter=200,
        theta_maxiter=200,
        psi_maxiter=200,
        outer_tol=1e-5,
        rho_init=0.1,
        save_excel=True,
        xlsx_name="mae_results.xlsx"
):
    """
    Repeated simulation output:
    1. Absolute errors (AE) for rho, beta_j, S, and R in each replication
    """

    # Store absolute errors from each replication
    rho_mae_list = []
    beta_mae_list = [[] for _ in range(p)]
    S_mae_list = []
    R_mae_list = []
    records = []

    # Run Monte Carlo replications
    for rep in range(n_rep):
        seed = seed_base + rep

        # Initialize parameters
        init_params = np.zeros(2 * n + p, dtype=float)
        init_params[2 * n - 1] = rho_init

        # Simulate data
        Y, Z, X, true_params = simulate_dynamic_ordinal_poisson_data(
            n=n,
            T=T,
            p=p,
            alphas=alphas,
            seed=seed
        )

        # Fit the model
        fit_res = fit_dynamic_ordinal_poisson_two_step(
            Y=Y,
            X=X,
            alphas=alphas,
            init_params=init_params,
            outer_maxiter=outer_maxiter,
            theta_maxiter=theta_maxiter,
            psi_maxiter=psi_maxiter,
            outer_tol=outer_tol
        )

        # Extract convergence diagnostics for theta
        theta_success = fit_res["theta_result"]["success"]
        theta_message = fit_res["theta_result"]["message"]
        theta_n_iter = fit_res["theta_result"]["n_iter"]

        # Extract true and estimated parameters
        rho_true = true_params["rho_true"]
        beta_true = true_params["beta_true"]
        S_true = true_params["S_true"]
        R_true = true_params["R_true"]
        print(f"R_true: {R_true}")
        rho_hat = fit_res["rho_hat"]
        beta_hat = fit_res["beta_hat"]
        S_hat = fit_res["S_hat"]
        R_hat = fit_res["R_hat"]
        print(f"R_hat: {R_hat}")
  # Implementation note.
        rho_mae = np.abs(rho_hat - rho_true)

        beta_mae = np.abs(beta_hat - beta_true)  # shape=(p,)
        S_mae = np.mean(np.abs(S_hat - S_true))
        R_mae = np.mean(np.abs(R_hat[:n - 1] - R_true[:n - 1]))

        # Store results
        rho_mae_list.append(rho_mae)
        for j in range(p):
            beta_mae_list[j].append(beta_mae[j])
        S_mae_list.append(S_mae)
        R_mae_list.append(R_mae)

        # Print one summary line after each replication
        line_parts = [
            f"Rep {rep + 1:3d}",
            f"rho_hat={rho_hat:.6f}",
            f"rho_mae={rho_mae:.6f}"
        ]

        for j in range(p):
            line_parts.append(f"beta{j + 1}_hat={beta_hat[j]:.6f}")
            line_parts.append(f"beta{j + 1}_mae={beta_mae[j]:.6f}")

        line_parts.append(f"S_mae={S_mae:.6f}")
        line_parts.append(f"R_mae={R_mae:.6f}")
        line_parts.append(f"S_hat_mean={np.mean(S_hat):.6f}")
        line_parts.append(f"R_hat_mean={np.mean(R_hat[:n - 1]):.6f}")
        line_parts.append(f"theta_success={theta_success}")
        line_parts.append(f"theta_msg={theta_message}")

        print(" | ".join(line_parts))

        # Record results from the current replication
        row = {
            "rep": rep + 1,

            "rho_hat": rho_hat,
            "rho_mae": rho_mae,

            "S_mae": S_mae,
            "R_mae": R_mae,
            "S_hat_mean": np.mean(S_hat),
            "R_hat_mean": np.mean(R_hat[:n - 1]),

            "theta_success": fit_res["theta_result"]["success"],
            "theta_message": fit_res["theta_result"]["message"],
            "theta_n_iter": fit_res["theta_result"]["n_iter"],
            "theta_final_nll": fit_res["theta_result"]["final_nll"],
            "theta_final_diff": fit_res["theta_result"]["final_diff"],

            "psi_success": fit_res["psi_result"].success,
            "psi_message": fit_res["psi_result"].message,
            "psi_n_iter": fit_res["psi_result"].nit,
            "psi_final_nll": fit_res["psi_result"].fun
        }
        for j in range(p):
            row[f"beta{j + 1}_hat"] = beta_hat[j]
            row[f"beta{j + 1}_mae"] = beta_mae[j]

        records.append(row)

    # Save all replication-level results
    if save_excel:
        df = pd.DataFrame(records)
        df.to_excel(xlsx_name, index=False)
        print(f"\nMonte Carlo results saved to: {xlsx_name}")

    return {
        "rho_mae_list": rho_mae_list,
        "beta_mae_list": beta_mae_list,
        "S_mae_list": S_mae_list,
        "R_mae_list": R_mae_list,
    }


if __name__ == "__main__":
    n = 10
    T = 6
    p = 3
    alphas = np.array([-1, 0, 2, np.inf], dtype=float)

    # ===== Portable output path =====
    save_path = "mae_results.xlsx"

    print("Starting Monte Carlo simulation...")

    mc_res = monte_carlo_dynamic_ordinal_poisson(
        n=n,
        T=T,
        p=p,
        alphas=alphas,
        n_rep=1,  # Number of replications
        seed_base=2026,  # Base random seed
        outer_maxiter=200,
        theta_maxiter=200,
        psi_maxiter=200,
        outer_tol=1e-5,
        rho_init=0.1,
        save_excel=True,
        xlsx_name=save_path
    )

    print("\nSimulation completed.")
