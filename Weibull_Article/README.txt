================================================================================
RELIABILITY ANALYSIS: INVERSE WEIBULL & PIECEWISE MODELS WITH OUTLIERS
================================================================================

This repository contains a Python-based framework designed for the simulation, 
estimation, and statistical validation of life-stress models. The primary 
focus is the study of the Inverse Weibull distribution and its robustness 
when facing contaminated populations (outliers).

--------------------------------------------------------------------------------
FILE DESCRIPTIONS
--------------------------------------------------------------------------------

1. CORE SIMULATION & PROBABILITY
   - simulation.py: 
     The primary engine for generating lifetime data. It implements Piecewise 
     Weibull models for Phase 1 & 2 stress levels and manages the 
     generation of outlier-contaminated populations.

   - Prop_outliers.py: 
     Specific scripts to define and manage the proportions and 
     characteristics of outliers within the samples.

2. PARAMETER ESTIMATION
   - MLE_estimation.py: 
     Implements Maximum Likelihood Estimation (MLE) for model parameters.
   - estimation.py: 
     General estimation algorithms and robust methods designed to 
     mitigate the impact of outliers on model accuracy.
   - auxiliarfunctions.py: 
     Support mathematics, for obtaining the variance and covariance.
3. CONFIDENCE INTERVALS & VARIABILITY
   - Obtain_Intervals.py: 
     Calculates variance-covariance matrices and generates confidence 
     intervals for the estimators.

4. SIMULATION OF A PARTICULAR TEST
   - ObservationsEstimationsCIInvWeib.py: 
     Simulates under specified conditions
5. EVALUATION & VISUALIZATION (MSE)
   - Observe_MSE_CI_Inv_Weib.py: 
     Analyzes the Root Mean Square Error (RMSE) of estimators under 
     various contamination levels (Beta and Proportion).
   - ObserveElipsoidsInvWeib.py: 
     Generates geometric visualizations (confidence ellipsoids) to 
     analyze the joint variability and correlation of estimated parameters.
   - Table_Confidence_IntervalInvWeib.py: 
     Specialized version for the Inverse Weibull model. It calculates 
     coverage using Logit and Log transformations to ensure intervals 
     remain within valid physical bounds.
6. REAL WORLD APPLICATION
   - RealData.py: 
     Dedicated script for processing and fitting the developed models 
     onto experimental or historical real-world datasets.

--------------------------------------------------------------------------------
TYPICAL WORKFLOW
--------------------------------------------------------------------------------

1. CONFIGURATION: Define true parameters and outlier types in simulation.py.
2. EXECUTION: Run Monte Carlo simulations to obtain mass estimations.
3. ANALYSIS: Calculate RMSE and coverage for the confidence intervals.
4. EXPORT: Results are typically saved as 'ResultsMSE_...xlsx' for final 
   plotting and article inclusion.

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
- Python 3.x
- NumPy, Pandas, Matplotlib, SciPy
================================================================================