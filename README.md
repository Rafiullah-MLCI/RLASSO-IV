# RLASSO-IV
An R function to implement an R-learner with Instrumental Variable for heterogeneous effect estimation.

# Details
**Function**: RLASSO-IV
**Type**: R Function
**Title**: Heterogeneous Treatment Effects Estimation using R-Learner with Instrumental Variables
**Version**: 1.0
**Date**: 2026-02-20
**Author**: Rafiullah
**Maintainer**: Rafiullah <rafiuom111@gmail.com>
**Description**: The RLASSO-IV function implements an instrumental-variable extension of the R-learner framework for estimating heterogeneous treatment effects under endogenous treatment assignment. It estimates nuisance functions q_hat(X)=E[Y|X], p_hat(X)=E[T|X], and h_hat(Z,X)=E[T|Z,X] using Support Vector Machines (SVM), constructs the instrument-induced residual T_tilde = h_hat(Z,X) - p_hat(X), and then estimates the heterogeneous IV effect via a sparse LASSO second stage (glmnet). The procedure supports cross-fitting to reduce overfitting bias, cross-validation for hyperparameter tuning, and reports first-stage diagnostics (e.g., partial F-statistic) for instrument strength. The method is suitable for moderate sample sizes and high-dimensional covariate settings.
**License**: GPL-3
**Depends**: R (>= 3.5.0)
**Suggested Packages**: e1071, glmnet, caret, Matrix
Encoding: UTF-8

## Usage
**Example data**
set.seed(123)
n <- 500
p <- 20

x <- matrix(rnorm(n*p), n, p)                 # Covariates
z <- rbinom(n, 1, plogis(0.5*x[,1]))          # Instrument
u <- rnorm(n)                                 # Unobserved confounder

w <- rbinom(n, 1, plogis(0.7*z + 0.3*x[,2] + 0.5*u))   # Endogenous treatment
tau_true <- 0.5 + 0.5*(x[,1] > 0)                       # Treatment effect
y <- tau_true*t + x[,3] + u + rnorm(n)                  # Outcome

**Fit RLASSO-IV**
rlasso_iv_fit <- rlasso_iv(x = x, z = z, w = w, y = y, K = 5)

**Predict treatment effects**
tau_hat <- predict(rlasso_iv_fit, newx = x)




# RLASSO-SVM
An R function to implement Residualized LASSO and Support Vector Machines (SVM) for treatment effect estimation.
## Details
**Function**: RLASSO-SVM  
**Type**: R Function  
**Title**: Residualized LASSO with Support Vector Machines for Heterogeneous Treatment Effects Estimation  
**Version**: 1.0  
**Date**: 2025-09-29  
**Author**: Rafullah
**Maintainer**: Rafullah <rafiuom111@gmail.com>  
**Description**: The RLASSO-SVM function implements a hybrid model combining Residualized Least Absolute Shrinkage and Selection Operator (LASSO) with Support Vector Machines (SVM) for estimating heterogeneous treatment effects estimation. It uses LASSO (via glmnet) for feature selection and SVM (via e1071) for estimating nuisance parameters (m_hat, p_hat), with preprocessing handled by caret and stringr. The function supports cross-validation, standardization, and an optional robust specification (rs) for high-dimensional datasets.  
**License**: GPL-3  
**Depends**: R (>= 3.5.0)  
**Suggested Packages**: caret, stringr, e1071, glmnet  
**Encoding**: UTF-8  

## Usage
```R
# Example data
set.seed(123)
n <- 500
p <- 20

x = matrix(rnorm(n*p), n, p) # Covariates
w = rbinom(n, 1, 0.5) # Treatment
y = pmax(x[,1], 0) * w + x[,2] + pmin(x[,3], 0) + rnorm(n) # Outcome
# Fit the model
Rlasso_svm <- rlasso_svm(x = x, w = w, y = y)

# Predict treatment effects
rlasso_svm_pred <- predict(Rlasso_svm, newx = x)
print(rlasso_svm_pred)

print(summary(tau_hat))

