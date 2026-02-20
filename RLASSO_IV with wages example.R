######################################
#Wages Example in the Paper          #
######################################

rm(list = ls())

# Libraries
library(tidyverse)
library(grf)
library(rlearner)
library(randomForest)
library(FNN)
library(MASS)
library(e1071)
library(rpart)
library(rpart.plot)
# Functions
BScore <- function(x,y) mean((x-y)^2)
bias <- function(x, y) mean(x - y)
PEHE <- function(x, y) sqrt(mean((x - y)^2))
MSE <- function(x, y) mean((x - y)^2)

MC_se <- function(x, B) qt(0.975, B - 1) * sd(x) / sqrt(B)

## ===============================
## Card Wages: Semi-synthetic DGP
## File: /mnt/data/wages_return_data.csv
## ===============================

set.seed(42)

## 1) Load data
data<- read.csv(file.choose(), header = TRUE)
data <- na.omit(data)


## 2) Define key variables
y_obs <- data$lwage              # original outcome (keep for reference only)
T_edu <- data$educ               # years of education
w <- ifelse(T_edu >= 14, 1L, 0L) # binary treatment (your rule)
z <- data$nearc4                 # instrument

data$w <- w
data$z <- z

## 3) Basic sanity checks
stopifnot(!anyNA(data$educ), !anyNA(data$nearc4))

## 4) Define baseline covariates used in mu0(X) and tau(X)
##    (These are pre-treatment covariates; keep it interpretable for reviewers.)
exper_sqrt  <- sqrt(pmax(data$exper, 0))
parents_log <- log1p(pmax(data$fatheduc, 0) + pmax(data$motheduc, 0))
iq_scaled   <- data$iq / 10
age         <- data$age
black       <- data$black
south       <- data$south

## 5) Generate "true" heterogeneous treatment effect tau_true = tau(X)
##    IMPORTANT: tau_true depends ONLY on baseline X (not on educ or w or z).
u <- rnorm(nrow(data), mean = 0, sd = 0.2)

tau_true <-  0.3 * exper_sqrt +
  0.2 * parents_log +
  0.1 * iq_scaled -
  0.05 * age +
  0.2 * black -
  0.1 * south +
  u

data$tau_true <- tau_true
ITE = tau_true
## 6) Baseline outcome model mu0(X)
mu0 <- 1 +
  0.2 * exper_sqrt +
  0.1 * parents_log +
  0.05 * iq_scaled -
  0.02 * age +
  0.1 * black -
  0.05 * south

data$mu0 <- mu0

## 7) Generate potential outcomes and observed semi-synthetic outcome
eps <- rnorm(nrow(data), mean = 0, sd = 0.5)

y0 <- mu0 + eps
y1 <- mu0 + tau_true + eps
y_syn <- w * y1 + (1 - w) * y0

data$y0 <- y0
data$y1 <- y1
data$y_syn <- y_syn

y = y_syn
## 8) First-stage relevance check: Z -> W
##    (Simple difference in means + regression; include in appendix if needed.)
cat("\n--- First-stage relevance check (Z=nearc4 -> W) ---\n")
cat("Mean(W | Z=1):", mean(w[z == 1], na.rm = TRUE), "\n")
cat("Mean(W | Z=0):", mean(w[z == 0], na.rm = TRUE), "\n")
cat("Diff:", mean(w[z == 1], na.rm = TRUE) - mean(w[z == 0], na.rm = TRUE), "\n")

fs_lm <- lm(w ~ z)
cat("\nFirst-stage regression: W ~ Z\n")
print(summary(fs_lm))

## 9) Build covariate matrix for learners (exclude outcomes/treatment/instrument/constructed truths)
##    You can adjust this list as needed, but DO NOT include y0/y1/y_syn/tau_true.
exclude_cols <- c( "lwage","educ","w","nearc4","z","y0","y1","y_syn","tau_true","mu0","ITE")
keep_cols <- setdiff(colnames(data), exclude_cols)
x <- as.matrix(data[, keep_cols])

## 10) Final objects you will use in training/evaluation
y <- data$lwage
cat("\n--- Objects created ---\n")
cat("y (semi-synthetic outcome): length =", length(y), "\n")
cat("w (binary treatment):       mean   =", mean(w), "\n")
cat("z (instrument):             mean   =", mean(z), "\n")
cat("x (covariates):             dim    =", paste(dim(x), collapse=" x "), "\n")
cat("tau_true (true ITE):        mean   =", mean(data$tau_true), "\n")


# Verify dimensions
print(dim(x))  # Should show [n_samples, n_features]
print(class(w))  # Should be "numeric" (0/1)
### INFO
N <- 1600
P <- 25  
B <- 500

# Store Metrics
Bias_XLasso <- Bias_ULasso <- Bias_RLASSO  <- Bias_SLasso <- Bias_TLasso <- Bias_CF <- Bias_IVCF <- rep(NA, B)
PEHE_XLasso <- PEHE_ULasso <-  PEHE_RLASSO  <- PEHE_SLasso <- PEHE_TLasso <- PEHE_CF <- PEHE_IVCF <- rep(NA, B)
PEHE_RLASSO_IV <- Bias_RLASSO_IV <- rep(NA, B)


for (b in 1:B) {
  cat("\n\n\n\n*** Iteration ", b, "\n\n\n")
  set.seed(b * 50)
  
  n = 1600
  ###### MODELS ESTIMATION  ------------------------------------------------
  
  
  sanitize_x = function(x){
    # make sure x is a numeric matrix with named columns (for caret)
    if (!is.matrix(x) || !is.numeric(x) || any(is.na(x))) {
      stop("x must be a numeric matrix with no missing values")
    }
    colnames(x) = stringr::str_c("covariate_", 1:ncol(x))
    return(x)
  }
  
  sanitize_input = function(x, z, w, y) {
    x = sanitize_x(x)
    
    # check z
    if (!is.numeric(z) && !is.matrix(z) && !is.data.frame(z)) {
      stop("z must be a numeric vector, matrix, or data frame")
    }
    if (nrow(as.data.frame(z)) != nrow(x)) {
      stop("z must have the same number of rows as x")
    }
    
    # check w
    if (!is.numeric(w)) {
      stop("the input w should be a numeric vector")
    }
    if (is.numeric(w) && all(w %in% c(0,1))) {
      w = w == 1
    }
    
    # check y
    if (!is.numeric(y)) {
      stop("y should be a numeric vector")
    }
    
    # check dimensions
    if (length(y) != nrow(x) || length(w) != nrow(x)) {
      stop("nrow(x), length(w), and length(y) should all be equal")
    }
    
    return(list(x = x, z = z, w = w, y = y))
  }
  
  
  #############################################################
  ######   RLASSO_IV Function     ############################
  ############################################################
  
  rlasso_iv <- function(x, z, w, y,
                        alpha = 1,
                        k_folds = NULL,
                        foldid = NULL,
                        lambda_y = NULL,
                        lambda_w = NULL,
                        lambda_tau = NULL,
                        lambda_choice = c("lambda.min", "lambda.1se"),
                        rs = FALSE,
                        penalty_factor = NULL,
                        kernel = NULL,
                        type = NULL,
                        gamma = NULL,
                        cost = NULL
  ) {
    
    
    input = sanitize_input(x, z, w, y)
    x = input$x
    z = input$z
    w = input$w
    y = input$y
    
    standardization = caret::preProcess(x, method = c("center", "scale"))
    x_scl = predict(standardization, x)
    x_scl = x_scl[, !is.na(colSums(x_scl)), drop = FALSE]
    
    lambda_choice = match.arg(lambda_choice)
    nobs = nrow(x_scl); pobs = ncol(x_scl)
    
    if (is.null(foldid) || length(foldid) != length(w)) {
      if (!is.null(foldid) && length(foldid) != length(w)) warning("supplied foldid does not have the same length")
      if (is.null(k_folds)) k_folds = floor(max(3, min(10, length(w)/4)))
      foldid = sample(rep(seq(k_folds), length = length(w)))
    }
    
    if (is.null(penalty_factor) || length(penalty_factor) != pobs) {
      if (!is.null(penalty_factor)) warning("penalty_factor length mismatch; using all ones")
      penalty_factor_nuisance = rep(1, pobs)
    } else {
      penalty_factor_nuisance = penalty_factor
    }
    
    # Estimate CBPS-based propensity score e_hat
    
    
    e_fit = svm(w ~ ., data = cbind(w, x_scl), type= type, kernel=  kernel, cost= cost, gamma = gamma)
    e_hat = predict(e_fit, x_scl)
    # Append e_hat to x
    x_aug <- cbind(x_scl, e_hat)
    pobs_aug <- ncol(x_aug)
    
    penalty_factor_aug = c(penalty_factor_nuisance, 1)  # +1 for e_hat
    penalty_factor_tau = if (rs) c(0, penalty_factor_aug, penalty_factor_aug) else c(0, penalty_factor_aug)
    
    # First-stage: E[Y|X]
    y_fit = svm(y ~ ., data = cbind(y, x_aug), type= type, kernel= kernel, cost= cost, gamma = gamma)
    m_hat = predict(y_fit, x_aug)
    # First-stage: E[W|Z,X]
    wiv = cbind(z, x_aug)
    
    w_fit = svm(w ~ ., data = cbind(w, wiv), type= type, kernel= kernel, cost= cost, gamma = gamma)
    p_hat = predict(w_fit, wiv)
    # Robinson residuals
    y_tilde = y - m_hat
    w_tilde = p_hat - e_hat
    
    if (rs) {
      x_scl_tilde = cbind(as.numeric(w_tilde) * cbind(1, x_aug), x_aug)
      x_scl_pred = cbind(1, x_aug, x_aug * 0)
    } else {
      x_scl_tilde = cbind(as.numeric(w_tilde) * cbind(1, x_aug))
      x_scl_pred = cbind(1, x_aug)
    }
    
    # Second-stage: Estimate tau(x)
    tau_fit = glmnet::cv.glmnet(x_scl_tilde, y_tilde, foldid = foldid, alpha = alpha,
                                lambda = lambda_tau, penalty.factor = penalty_factor_tau,
                                standardize = FALSE)
    tau_beta = as.vector(t(coef(tau_fit, s = lambda_choice)[-1]))
    tau_hat = x_scl_pred %*% tau_beta
    
    ret = list(tau_fit = tau_fit,
               tau_beta = tau_beta,
               w_fit = w_fit,
               y_fit = y_fit,
               p_hat = p_hat,
               m_hat = m_hat,
               tau_hat = tau_hat,
               e_hat = e_hat,
               rs = rs,
               standardization = standardization,
               e_fit = e_fit)
    class(ret) <- "rlasso_iv"
    ret
  }
  
  predict.rlasso_iv <- function(object, newx = NULL, ...) {
    if (!is.null(newx)) {
      newx = sanitize_x(newx)
      newx_scl = predict(object$standardization, newx)
      newx_scl = newx_scl[, !is.na(colSums(newx_scl)), drop = FALSE]
      
      # Predict e_hat using the trained SVM model
      new_e_hat <- predict(object$e_fit, newdata = newx_scl)
      newx_aug = cbind(newx_scl, new_e_hat)
      
      newx_scl_pred = if (object$rs) {
        cbind(1, newx_aug, newx_aug * 0)
      } else {
        cbind(1, newx_aug)
      }
      
      tau_hat = newx_scl_pred %*% object$tau_beta
    } else {
      tau_hat = object$tau_hat
    }
    return(tau_hat)
  }

  
  ######################### RLASSO_IV
  rlasso_iv <- rlasso_iv(x = x, z = z, w = w, y = y, type = "eps-regression", kernel = "sigmoid", cost = 10, gamma = 0.01  )
  rlasso_iv_pred <- predict(rlasso_iv, x = x)
  ##### R-LASSO
  
  Rlasso <- rlasso(x = x, w = w, y = y)
  rlasso_pred <- predict(Rlasso, x = x)
  
  #######S-Lasso
  SLasso <- slasso(x = x, w = w, y = y)
  SLasso_pred <- predict(SLasso, x = x)
  
  ######################### T LASSO
  TLasso <- tlasso(x = x, w = w, y = y)
  TLasso_pred <- predict(TLasso, x = x)
  
  ######################### X LASSO
  XLasso <- xlasso(x = x, w = w, y = y)
  XLasso_pred <- predict(XLasso, x)
  
  ######################### U LASSO
  ULasso <- ulasso(x = x, w = w, y = y)
  ULasso_pred <- predict(ULasso, x)
  
  ######################### Causal forest
  CF <- causal_forest(X = x, Y = y, W = w)
  
  iv.forest <- instrumental_forest(
    x, y, w, z, 
    num.trees = 2000,
    sample.fraction = 0.30,   # MUST be < 0.5 when CI/variance enabled
    mtry = min(ceiling(sqrt(ncol(x)) + 20), ncol(x)),
    min.node.size = 10,
    honesty = TRUE,
    honesty.fraction = 0.5,
    ci.group.size = 2
  )
  
  Tau_IVCF <- iv.forest$predictions
  
  
  # Compute metrics
  
  Tau_RLASSO_IV <- rlasso_iv_pred
  Tau_RLASSO <- rlasso_pred
  Tau_SLasso <- SLasso_pred
  Tau_TLasso <- TLasso_pred
  Tau_XLasso <- XLasso_pred
  Tau_ULasso <- ULasso_pred
  Tau_CF <- CF$predictions
  
  
  Bias_RLASSO[b] <- bias(Tau_RLASSO, ITE)
  Bias_RLASSO_IV[b] <- bias(Tau_RLASSO_IV, ITE)
  Bias_SLasso[b] <- bias(Tau_SLasso, ITE)
  Bias_TLasso[b] <- bias(Tau_TLasso, ITE)
  Bias_XLasso[b] <- bias(Tau_XLasso, ITE)
  Bias_ULasso[b] <- bias(Tau_ULasso, ITE)
  Bias_CF[b] <- bias(Tau_CF, ITE)
  Bias_IVCF[b] <- bias(Tau_IVCF, ITE)
  
  
  PEHE_RLASSO[b] <- PEHE(Tau_RLASSO, ITE)
  PEHE_RLASSO_IV[b] <- PEHE(Tau_RLASSO_IV, ITE)
  PEHE_SLasso[b] <- PEHE(Tau_SLasso, ITE)
  PEHE_TLasso[b] <- PEHE(Tau_TLasso, ITE)
  PEHE_XLasso[b] <- PEHE(Tau_XLasso, ITE)
  PEHE_ULasso[b] <- PEHE(Tau_ULasso, ITE)
  PEHE_CF[b] <- PEHE(Tau_CF, ITE)
  PEHE_IVCF[b] <- PEHE(Tau_IVCF, ITE)
  
}

# Final Metrics
Bias_Final <- data.frame(
  
  RLASSO = c(mean(Bias_RLASSO), MC_se(Bias_RLASSO, B)),
  RLASSO_IV = c(mean(Bias_RLASSO_IV), MC_se(Bias_RLASSO_IV, B)),
  SLASSO = c(mean(Bias_SLasso), MC_se(Bias_SLasso, B)),
  TLASSO = c(mean(Bias_TLasso), MC_se(Bias_TLasso, B)),
  XLASSO = c(mean(Bias_XLasso), MC_se(Bias_XLasso, B)),
  ULASSO = c(mean(Bias_ULasso), MC_se(Bias_ULasso, B)),
  CF = c(mean(Bias_CF), MC_se(Bias_CF, B)),
  IV_GRF = c(mean(Bias_IVCF), MC_se(Bias_IVCF, B))
  
)
rownames(Bias_Final) <- c("Bias", "SE")

PEHE_Final <- data.frame(
  
  RLASSO = c(mean(PEHE_RLASSO), MC_se(PEHE_RLASSO, B)),
  RLASSO_IV = c(mean(PEHE_RLASSO_IV), MC_se(PEHE_RLASSO_IV, B)),
  SLASSO = c(mean(PEHE_SLasso), MC_se(PEHE_SLasso, B)),
  TLASSO = c(mean(PEHE_TLasso), MC_se(PEHE_TLasso, B)),
  XLASSO = c(mean(PEHE_XLasso), MC_se(PEHE_XLasso, B)),
  ULASSO = c(mean(PEHE_ULasso), MC_se(PEHE_ULasso, B)),
  CF = c(mean(PEHE_CF), MC_se(PEHE_CF, B)),
  IV_GRF = c(mean(PEHE_IVCF), MC_se(PEHE_IVCF, B))
)
rownames(PEHE_Final) <- c("PEHE", "SE")
# All iterations results
PEHE_Single = data.frame(
  
  RLASSO = PEHE_RLASSO,
  RLASSO_IV = PEHE_RLASSO_IV,
  SLASSO = PEHE_SLasso,
  TLASSO = PEHE_TLasso, 
  XLASSO = PEHE_XLasso,
  ULASSO = PEHE_ULasso,
  CRF = PEHE_CF,
  IV_GRF = PEHE_IVCF
  
)

Bias_Single = data.frame(
  
  RLASSO = Bias_RLASSO, 
  RLASSO_IV = Bias_RLASSO_IV,
  SLASSO = Bias_SLasso, 
  TLASSO = Bias_TLasso,
  XLASSO = Bias_XLasso, 
  ULASSO = Bias_ULasso,
  CRF = Bias_CF,
  IV_GRF = Bias_IVCF
  
)

# Print Results
Bias_Final
PEHE_Final




