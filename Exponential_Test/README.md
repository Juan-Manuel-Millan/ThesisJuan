========================================================================
PROJECT README: ROBUST INFERENCE IN STEP-STRESS ACCELERATED LIFE TESTING
========================================================================

This repository contains the numerical implementation, Monte Carlo 
simulation pipelines, and visualization routines for robust estimation 
and hypothesis testing (Z-type and Rao-type tests) based on the Minimum 
Density Power Divergence Estimator (MDPDE / RMDPDE) under a simple 
step-stress accelerated life testing model with exponential lifetimes.

The scripts are organized into three functional layers:
  1. Core Theoretical & Algorithmic Modules (Estimation & Testing)
  2. Data Generation & Monte Carlo Simulation Pipelines
  3. Visualization & Empirical Analysis (Data Inspection & Plotting)

------------------------------------------------------------------------
1. CORE MODULES (ESTIMATION, SIMULATION & STATISTICAL TESTS)
------------------------------------------------------------------------
These files define the fundamental mathematical tools, closed-form 
matrices, divergence functions, and simulation engines:

* simulation.py
  - Purpose: Simulates failure and right-censored times under a cumulative 
    exposure step-stress model with two stress levels (piecewise exponential).
  - Features: Implements inverse transform sampling for regular observations 
    and handles contamination (outliers) by sampling from a truncated 
    exponential distribution on a specified interval [t_start, t_end].

* estimation.py
  - Purpose: Core numerical optimization engine for parameter estimation.
  - Features: 
    - Closed-form unconstrained Maximum Likelihood Estimator (MLE).
    - Constrained MLE subject to linear constraints (M*a = d) and 
      order restrictions via SLSQP.
    - Density power divergence (beta-distance) minimization for beta >= 0, 
      yielding the MDPDE and RMDPDE.

* obtain_intervals.py / theoretical_matrices.py
  - Purpose: Computes the theoretical asymptotic variance-covariance 
    matrices for the density power divergence estimators.
  - Features: Calculates the deterministic components of the sensitivity 
    matrix J_beta and variability matrix K_beta, as well as the cross-terms 
    Xi_beta and sandwich covariance J^{-1} K J^{-1}.

* obtain_test.py
  - Purpose: Constructs the robust test statistics for linear hypothesis 
    testing H0: M*a = d.
  - Features:
    - Z-type test statistics (scalar and general Wald-type formulations).
    - Robust Rao-type (score-type) test statistics using the empirical 
      score vector U_{beta,N}(a) and projected matrices Q_beta.

------------------------------------------------------------------------
2. DATA GENERATION PIPELINES (MONTE CARLO SIMULATION DRIVERS)
------------------------------------------------------------------------
These standalone execution scripts run large-scale simulation loops and 
export the raw Monte Carlo results to Excel spreadsheets (.xlsx):

* run_simulations_rmdpde.py (main execution pipeline)
  - Purpose: Evaluates estimator performance across different contamination 
    rates (epsilon) and tuning parameters (beta).
  - Workflow: Iterates over 1,000 Monte Carlo runs per scenario, injecting 
    controlled outliers into either a0 or a1. Fits unrestricted and 
    restricted models and exports parameter estimates and outlier proportions 
    to files such as 'Results_RMDPDE_a1true_dtrue.xlsx'.

* run_tests_num_simulations.py / generate_test_data.py
  - Purpose: Assesses test sensitivity as a function of sample size (N) 
    under the null hypothesis H0 (empirical level) and alternative H1 
    (empirical power).
  - Output: Exports detailed logs containing test statistic values and 
    empirical rejections for both Z-type and Rao-type tests across sample 
    sizes.

------------------------------------------------------------------------
3. VISUALIZATION & ANALYSIS SCRIPTS (DATA INSPECTION & PLOTTING)
------------------------------------------------------------------------
These scripts process the generated Excel datasets to produce publication-
quality figures (Matplotlib) and summary tables (LaTeX):

* plot_rmse_efficiency.py
  - Purpose: Reads simulation output to evaluate point estimator accuracy.
  - Plots:
    - Root Mean Squared Error (RMSE) of a0 as a function of contamination 
      rate epsilon.
    - Relative efficiency gain/loss compared to the classical MLE (beta = 0).
    - Contrasts well-specified vs. misspecified baseline models.

* plot_empirical_level_power_ztype.py
  - Purpose: Analyzes the empirical level and power of the Z-type test.
  - Plots: Rejection proportion curves against contamination rate epsilon 
    and sample size N, comparing robustness of MDPDE vs. MLE.

* plot_empirical_level_power_rao.py
  - Purpose: Analyzes the empirical performance of the Rao-type score test.
  - Plots: Empirical level (at nominal alpha = 0.05) and empirical power 
    curves under outlier contamination in a0 and a1.

* build_results_table.py
  - Purpose: Processes real or single-dataset lifetime records across 
    multiple beta values.
  - Output: Prints a formatted summary table and generates LaTeX tabular 
    code displaying unrestricted estimates, restricted estimates, test 
    statistics, and corresponding p-values.
========================================================================
