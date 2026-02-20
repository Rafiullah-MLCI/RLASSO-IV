###############################################################
# FULL CODE RLASSO-IV Simulation with 25 covariate
# 
###############################################################

rm(list = ls())

# ======================
# Libraries
# ======================
library(tidyverse)
library(MASS)
library(e1071)      # svm
library(glmnet)     # cv.glmnet
library(caret)      # preProcess
library(grf)        # causal_forest, instrumental_forest
library(rlearner)   # rlasso/slasso/tlasso/xlasso/ulasso
library(ggplot2)
library(reshape2)
library(dplyr)

# ======================
# Helper functions
# ======================
bias <- function(x, y) mean(x - y)
PEHE <- function(x, y) sqrt(mean((x - y)^2))
MSE  <- function(x, y) mean((x - y)^2)
MC_se <- function(x, B) qt(0.975, B - 1) * sd(x) / sqrt(B)
clamp01 <- function(p, eps = 0.01) pmin(pmax(p, eps), 1 - eps)

# Generate correlated covariates
get_features <- function(N, P) {
  mysigma <- matrix(1, P, P)
  for (i in 1:P) {
    for (j in 1:P) {
      mysigma[i, j] <- 0.7^abs(i - j) + ifelse(i == j, 0, 0.1)
    }
  }
  mycop <- MASS::mvrnorm(N, rep(0, P), Sigma = mysigma)
  unif <- pnorm(mycop)
  
  x <- matrix(NA, N, P)
  x[, 1:10] <- qnorm(unif[, 1:10])
  x[, 11:P] <- qbinom(unif[, 11:P], 1, 0.3)
  x
}

sanitize_x <- function(x){
  if (!is.matrix(x) || !is.numeric(x) || any(is.na(x))) {
    stop("x must be a numeric matrix with no missing values")
  }
  colnames(x) <- paste0("covariate_", 1:ncol(x))
  x
}

sanitize_input <- function(x, z, w, y) {
  x <- sanitize_x(x)
  
  if (!is.numeric(z) && !is.matrix(z) && !is.data.frame(z)) stop("z must be numeric")
  if (nrow(as.data.frame(z)) != nrow(x)) stop("z must have same nrow as x")
  
  if (!is.numeric(w)) stop("w must be numeric")
  if (is.numeric(w) && all(w %in% c(0,1))) w <- w == 1
  
  if (!is.numeric(y)) stop("y must be numeric")
  if (length(y) != nrow(x) || length(w) != nrow(x)) stop("dimension mismatch")
  
  list(x = x, z = z, w = w, y = y)
}

# ==========================================================
# RLASSO-IV Function
# ==========================================================
rlasso_iv <- function(x, z, w, y,
                      alpha = 1,
                      lambda_tau = NULL,
                      lambda_choice = c("lambda.min","lambda.1se"),
                      rs = FALSE,
                      trim_eps = 0.01,
                      penalty_factor = NULL,
                      type = NULL,
                      kernel = "radial",
                      cost = 1,
                      gamma = 0.01,
                      cross = 5) {
  
  input <- sanitize_input(x, z, w, y)
  x <- input$x
  z <- as.numeric(input$z)
  y <- as.numeric(input$y)
  
  # keep w as numeric 0/1 for regression; and factor for classification
  w_num <- if (is.logical(input$w)) as.numeric(input$w) else as.numeric(input$w)
  w_fac <- factor(as.integer(w_num), levels = c(0, 1))
  
  # standardize X
  standardization <- caret::preProcess(x, method = c("center", "scale"))
  x_scl <- predict(standardization, x)
  x_scl <- x_scl[, !is.na(colSums(x_scl)), drop = FALSE]
  
  pobs <- ncol(x_scl)
  lambda_choice <- match.arg(lambda_choice)
  
  if (is.null(penalty_factor) || length(penalty_factor) != pobs) {
    penalty_factor_nuisance <- rep(1, pobs)
  } else {
    penalty_factor_nuisance <- penalty_factor
  }
  
  # ---------------------------------------
  # 1) e(x) = P(W=1|X)  (classification)
  # ---------------------------------------
  e_fit <- svm(x_scl, w_fac,
               type = "C-classification",
               kernel = kernel, cost = cost, gamma = gamma,
               probability = TRUE)
  
  e_prob <- attr(predict(e_fit, x_scl, probability = TRUE), "probabilities")[, "1"]
  e_hat  <- clamp01(as.numeric(e_prob), trim_eps)
  
  # ---------------------------------------
  # AUGMENTATION: X_aug = [X, e_hat]
  # ---------------------------------------
  x_aug <- cbind(x_scl, e_hat = e_hat)
  
  # ---------------------------------------
  # 2) m(x) = E[Y | X]
  # ---------------------------------------
  y_fit <- svm(x = x_aug, y = y,
               type = "nu-regression",
               kernel = kernel, cost = cost, gamma = gamma,
               cross = cross)
  m_hat <- as.numeric(predict(y_fit, x_aug))
  
  # ---------------------------------------
  # 3) p(z,x) = P(W=1 | Z, X)
  # ---------------------------------------
  wiv <- cbind(z = z, x_aug)
  
  w_fit <- svm(x = wiv, y = w_fac,
               type = "C-classification",
               kernel = kernel, cost = cost, gamma = gamma,
               probability = TRUE)
  
  p_prob <- attr(predict(w_fit, wiv, probability = TRUE), "probabilities")[, "1"]
  p_hat  <- clamp01(as.numeric(p_prob), trim_eps)
  
  # ---------------------------------------
  # Residuals
  # ---------------------------------------
  y_tilde <- y - m_hat
  w_tilde <- p_hat - e_hat
  
  # ---------------------------------------
  # 4) Tau stage (glmnet) 
  # ---------------------------------------
  penalty_factor_aug <- c(penalty_factor_nuisance, 1)  # +1 for e_hat
  
  if (rs) {
    X_second <- cbind(as.numeric(w_tilde) * cbind(1, x_aug), x_aug)
    X_pred   <- cbind(1, x_aug, x_aug * 0)
    penalty_factor_tau <- c(0, penalty_factor_aug, penalty_factor_aug)
  } else {
    X_second <- cbind(as.numeric(w_tilde) * cbind(1, x_aug))
    X_pred   <- cbind(1, x_aug)
    penalty_factor_tau <- c(0, penalty_factor_aug)
  }
  
  tau_fit <- glmnet::cv.glmnet(
    x = X_second, y = y_tilde,
    alpha = alpha,
    lambda = lambda_tau,
    penalty.factor = penalty_factor_tau,
    standardize = FALSE
  )
  
  tau_beta <- as.vector(coef(tau_fit, s = lambda_choice))[-1]
  tau_hat  <- as.numeric(X_pred %*% tau_beta)
  
  ret <- list(
    tau_fit = tau_fit,
    tau_beta = tau_beta,
    tau_hat = tau_hat,
    rs = rs,
    standardization = standardization,
    e_fit = e_fit     # needed for predict() to build e_hat on new data
  )
  class(ret) <- "rlasso_iv"
  ret
}

# ==========================================================
# Predict method 
# ==========================================================
predict.rlasso_iv <- function(object, newx = NULL, ...) {
  if (is.null(newx)) return(object$tau_hat)
  
  newx <- sanitize_x(newx)
  newx_scl <- predict(object$standardization, newx)
  newx_scl <- newx_scl[, !is.na(colSums(newx_scl)), drop = FALSE]
  
  # predict e_hat on new data then augment
  new_e_prob <- attr(predict(object$e_fit, newx_scl, probability = TRUE), "probabilities")[, "1"]
  new_e_hat  <- clamp01(as.numeric(new_e_prob), 0.01)
  
  newx_aug <- cbind(newx_scl, e_hat = new_e_hat)
  
  if (object$rs) {
    newX_pred <- cbind(1, newx_aug, newx_aug * 0)
  } else {
    newX_pred <- cbind(1, newx_aug)
  }
  
  as.numeric(newX_pred %*% object$tau_beta)
}

# ======================
# Simulation settings
# ======================
N <- 1000   # N = 250, 500, 1000, 2000
P <- 25
B <- 1000

# Strength of unobserved confounding U
conf_w <- 1.0
conf_y <- 1.0

# Storage vectors
Bias_RLASSO_IV <- PEHE_RLASSO_IV <- rep(NA, B)  # RLASSO-IV stored here
Bias_RLASSO <- PEHE_RLASSO <- rep(NA, B)
Bias_SLasso <- PEHE_SLasso <- rep(NA, B)
Bias_TLasso <- PEHE_TLasso <- rep(NA, B)
Bias_XLasso <- PEHE_XLasso <- rep(NA, B)
Bias_ULasso <- PEHE_ULasso <- rep(NA, B)
Bias_CF  <- PEHE_CF <- rep(NA, B)
Bias_IVCF <- PEHE_IVCF <- rep(NA, B)
Bias_SOLS  <- PEHE_SOLS <- rep(NA, B)
Bias_TOLS <- PEHE_TOLS <- rep(NA, B)

# IV diagnostics
Fstat_IV <- rep(NA, B)
Pval_IV  <- rep(NA, B)

# Store last iteration for ITE plots
ITE_last <- NULL
rboost_pred_last <- NULL

# ======================
# Monte Carlo loop
# ======================
for (b in 1:B) {
  
  cat("\n\n*** Iteration", b, "***\n")
  set.seed(b * 50)
  
  x <- get_features(N, P)
  
  # Unobserved confounder U
  U <- rnorm(N, 0, 1)
  
  mu_base <- 2 + 0.5*sin(pi*x[, 1]) - 0.25*x[, 2]^2 + 0.75*x[, 3]*x[, 9]
  mu <- 5 * mu_base
  ITE <- 1 + 2 * abs(x[, 4]) + x[, 10]
  
  # Instrument depends only on X
  z <- rbinom(N, 1, prob = plogis(0.5 * x[, 1] - 0.25 * x[, 5] + 0.5))
  
  # Endogenous treatment: depends on Z, X, and U
  pW <- plogis(1.2 * z + 0.9 * mu_base + conf_w * U)
  w <- rbinom(N, 1, prob = pW)
  
  # Outcome depends on W, X, and U
  y <- mu + ITE * w + conf_y * U + rnorm(N, mean = 0, sd = sd(ITE) / 2)
  
  # =========================
  # First-stage F-test (LPM)
  # =========================
  X_df <- as.data.frame(x)
  colnames(X_df) <- paste0("X", 1:ncol(X_df))
  fs_full <- lm(w ~ z + ., data = cbind(w = w, z = z, X_df))
  fs_restricted <- lm(w ~ . - z, data = cbind(w = w, z = z, X_df))
  fs_test <- anova(fs_restricted, fs_full)
  
  Fstat_IV[b] <- fs_test$F[2]
  Pval_IV[b]  <- fs_test$`Pr(>F)`[2]
  cat("First-stage F:", Fstat_IV[b], " | p-value:", Pval_IV[b], "\n")
  
  # =========================
  # RLASSO-IV
  # =========================
  fit_rlassoiv <- rlasso_iv(
    x = x, z = z, w = w, y = y,
    type = "nu-regression", kernel = "radial",
    cost = 1, gamma = 0.01, cross = 5
  )
  rlasso_iv_pred <- as.numeric(predict(fit_rlassoiv, x))  # RLASSO-IV predicted ITE
  fit_rlassoiv$e_fit
  # =========================
  # Benchmarks
  # =========================
  fit_rlasso <- rlasso(x = x, w = w, y = y)
  rlasso_pred <- as.numeric(predict(fit_rlasso, newx = x))
  
  fit_slasso <- slasso(x = x, w = w, y = y)
  SLasso_pred <- as.numeric(predict(fit_slasso, x = x))
  
  fit_tlasso <- tlasso(x = x, w = w, y = y)
  TLasso_pred <- as.numeric(predict(fit_tlasso, x = x))
  
  fit_xlasso <- xlasso(x = x, w = w, y = y)
  XLasso_pred <- as.numeric(predict(fit_xlasso, x))
  
  fit_ulasso <- ulasso(x = x, w = w, y = y)
  ULasso_pred <- as.numeric(predict(fit_ulasso, x))
  
  CF <- causal_forest(X = x, Y = y, W = w)
  Tau_CF <- as.numeric(CF$predictions)
  
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
  
  # SOLS / TOLS
  y_train <- y
  z_train <- w
  train_augmX <- cbind(x)
  Train_ITE <- ITE
  
  SOLS <- lm(cbind.data.frame(y_train, z_train, train_augmX))
  Y0_train_SOLS <- predict(SOLS, newdata = cbind.data.frame(train_augmX, z_train = 0), se.fit = TRUE)
  Y1_train_SOLS <- predict(SOLS, newdata = cbind.data.frame(train_augmX, z_train = 1), se.fit = TRUE)
  Tau_SOLS <- as.numeric(Y1_train_SOLS$fit - Y0_train_SOLS$fit)
  
  TOLS1 <- lm(cbind.data.frame(y_train, z_train, train_augmX)[z_train == 1, ])
  TOLS0 <- lm(cbind.data.frame(y_train, z_train, train_augmX)[z_train == 0, ])
  Y0_train_TOLS <- predict(TOLS0, newdata = cbind.data.frame(train_augmX, z_train = 0), se.fit = TRUE)
  Y1_train_TOLS <- predict(TOLS1, newdata = cbind.data.frame(train_augmX, z_train = 1), se.fit = TRUE)
  Tau_TOLS <- as.numeric(Y1_train_TOLS$fit - Y0_train_TOLS$fit)
  
  # =========================
  # Metrics
  # =========================
  Bias_RLASSO_IV[b] <- bias(rlasso_iv_pred, ITE)
  PEHE_RLASSO_IV[b] <- PEHE(rlasso_iv_pred, ITE)
  
  Bias_RLASSO[b] <- bias(rlasso_pred, ITE)
  PEHE_RLASSO[b] <- PEHE(rlasso_pred, ITE)
  
  Bias_SLasso[b] <- bias(SLasso_pred, ITE)
  PEHE_SLasso[b] <- PEHE(SLasso_pred, ITE)
  
  Bias_TLasso[b] <- bias(TLasso_pred, ITE)
  
  PEHE_TLasso[b] <- PEHE(TLasso_pred, ITE)
  
  Bias_XLasso[b] <- bias(XLasso_pred, ITE)
  PEHE_XLasso[b] <- PEHE(XLasso_pred, ITE)
  
  Bias_ULasso[b] <- bias(ULasso_pred, ITE)
  PEHE_ULasso[b] <- PEHE(ULasso_pred, ITE)
  
  Bias_CF[b] <- bias(Tau_CF, ITE)
  PEHE_CF[b] <- PEHE(Tau_CF, ITE)
  
  Bias_IVCF[b] <- bias(Tau_IVCF, ITE)
  PEHE_IVCF[b] <- PEHE(Tau_IVCF, ITE)
  
  Bias_SOLS[b] <- bias(Tau_SOLS, ITE)
  PEHE_SOLS[b] <- PEHE(Train_ITE, Tau_SOLS)
  
  Bias_TOLS[b] <- bias(Tau_TOLS, ITE)
  PEHE_TOLS[b] <- PEHE(Train_ITE, Tau_TOLS)
  
  if (b == B) {
    ITE_last <- ITE
    rlasso_iv_pred_last <- rlasso_iv_pred
  }
}

# ======================
# FINAL TABLES (means + MC SE)  
# ======================
Bias_Final <- data.frame(
  RLASSO_IV = c(mean(Bias_RLASSO_IV), MC_se(Bias_RLASSO_IV, B)),
  RLASSO    = c(mean(Bias_RLASSO),     MC_se(Bias_RLASSO, B)),
  SLASSO    = c(mean(Bias_SLasso),    MC_se(Bias_SLasso, B)),
  TLASSO    = c(mean(Bias_TLasso),    MC_se(Bias_TLasso, B)),
  XLASSO    = c(mean(Bias_XLasso),    MC_se(Bias_XLasso, B)),
  ULASSO    = c(mean(Bias_ULasso),    MC_se(Bias_ULasso, B)),
  CRF       = c(mean(Bias_CF),        MC_se(Bias_CF, B)),
  IV_GRF    = c(mean(Bias_IVCF),      MC_se(Bias_IVCF, B)),
  SOLS      = c(mean(Bias_SOLS),      MC_se(Bias_SOLS, B)),
  TOLS      = c(mean(Bias_TOLS),      MC_se(Bias_TOLS, B))
)
rownames(Bias_Final) <- c("Bias", "SE")


PEHE_Final <- data.frame(
  RLASSO_IV = c(mean(PEHE_RLASSO_IV), MC_se(PEHE_RLASSO_IV, B)),
  RLASSO    = c(mean(PEHE_RLASSO),     MC_se(PEHE_RLASSO, B)),
  SLASSO    = c(mean(PEHE_SLasso),    MC_se(PEHE_SLasso, B)),
  TLASSO    = c(mean(PEHE_TLasso),    MC_se(PEHE_TLasso, B)),
  XLASSO    = c(mean(PEHE_XLasso),    MC_se(PEHE_XLasso, B)),
  ULASSO    = c(mean(PEHE_ULasso),    MC_se(PEHE_ULasso, B)),
  CRF       = c(mean(PEHE_CF),        MC_se(PEHE_CF, B)),
  IV_GRF    = c(mean(PEHE_IVCF),      MC_se(PEHE_IVCF, B)),
  SOLS      = c(mean(PEHE_SOLS),      MC_se(PEHE_SOLS, B)),
  TOLS      = c(mean(PEHE_TOLS),      MC_se(PEHE_TOLS, B))
)
rownames(PEHE_Final) <- c("PEHE", "SE")

cat("\n\n====================\nFINAL RESULTS\n====================\n")
cat("\nBias_Final:\n"); print(Bias_Final)
cat("\nPEHE_Final:\n"); print(PEHE_Final)

# ======================
# IV diagnostics summary
# ======================
cat("\n\n====================\nIV First-stage diagnostics\n====================\n")
cat("Mean F:", mean(Fstat_IV, na.rm = TRUE), "\n")
cat("Median F:", median(Fstat_IV, na.rm = TRUE), "\n")
cat("Share F>10:", mean(Fstat_IV > 10, na.rm = TRUE), "\n")
print(quantile(Fstat_IV, c(0.1, 0.25, 0.5, 0.75, 0.9), na.rm = TRUE))



# ======================
# ===== YOUR REQUESTED MODEL COMPARISON PLOTS =====
# ======================
PEHE_Single <- data.frame(
  RLASSO_IV = PEHE_RLASSO_IV,
  RLASSO = PEHE_RLASSO,
  SLASSO = PEHE_SLasso,
  TLASSO = PEHE_TLasso,
  XLASSO = PEHE_XLasso,
  ULASSO = PEHE_ULasso,
  CRF = PEHE_CF,
  IV_GRF = PEHE_IVCF,
  SOLS = PEHE_SOLS,
  TOLS = PEHE_TOLS
)

Bias_Single <- data.frame(
  RLASSO_IV = Bias_RLASSO_IV,
  RLASSO = Bias_RLASSO,
  SLASSO = Bias_SLasso,
  TLASSO = Bias_TLasso,
  XLASSO = Bias_XLasso,
  ULASSO = Bias_ULasso,
  CRF = Bias_CF,
  IV_GRF = Bias_IVCF,
  SOLS = Bias_SOLS,
  TOLS = Bias_TOLS
)



PEHE_Single$Iteration <- 1:nrow(PEHE_Single)
Bias_Single$Iteration <- 1:nrow(Bias_Single)

PEHE_long <- melt(PEHE_Single, id.vars = "Iteration", variable.name = "Model", value.name = "PEHE")
Bias_long <- melt(Bias_Single, id.vars = "Iteration", variable.name = "Model", value.name = "Bias")

PEHE_summary <- PEHE_long %>% group_by(Model) %>% summarise(mean_PEHE = mean(PEHE))
Bias_summary <- Bias_long %>% group_by(Model) %>% summarise(mean_Bias = mean(Bias))





# === LINE CHARTS (mean across models) ===

# PEHE Plot
ggplot(PEHE_summary, aes(x = reorder(Model, mean_PEHE), y = mean_PEHE, group = 1)) +
  geom_line(color = "black", size = 1) +  # same color line
  geom_point(aes(color = Model), size = 3) +  # colored dots
  labs(title = "Sample Size N = 2000, Number of Covariates P = 25",
       y = "Mean PEHE", x = "Model") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "italic", size = 12) ,
        axis.title.x = element_text(face = "bold"),
        axis.title.y = element_text(face = "bold")) +
  guides(color = "none")  # remove legend if you prefer

# Bias Plot
ggplot(Bias_summary, aes(x = reorder(Model, mean_Bias), y = mean_Bias, group = 1)) +
  geom_line(color = "black", size = 1) +
  geom_point(aes(color = Model), size = 3) +
  labs(title = "Sample Size N = 2000, Number of Covariates P = 25",
       y = "Mean Bias", x = "Model") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "italic", size = 12),
        axis.title.x = element_text(face = "bold"),
        axis.title.y = element_text(face = "bold")) +
  guides(color = "none")


# === BOXPLOTS ===
# Boxplot for PEHE
ggplot(PEHE_long, aes(x = Model, y = PEHE, fill = Model)) +
  geom_boxplot() + labs(title = "Sample Size N = 2000, Number of Covariates P = 25", y = "PEHE", x = "Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme_minimal() + theme(plot.title = element_text(face = "italic", size = 12)
                          ,
                          axis.title.x = element_text(face = "bold"),
                          axis.title.y = element_text(face = "bold"))

# Boxplot for Bias
ggplot(Bias_long, aes(x = Model, y = Bias, fill = Model)) +
  geom_boxplot() +labs(title = "Sample Size N = 2000, Number of Covariates P = 25", y = "Bias", x = "Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme_minimal() + theme(plot.title = element_text(face = "italic", size = 12),
                          axis.title.x = element_text(face = "bold"),
                          axis.title.y = element_text(face = "bold"))

PEHE_Line_Plot_1000_10

library(ggplot2)

# Violin + Boxplot
ggplot(Bias_long, aes(x = Model, y = Bias)) +
  geom_violin(fill = "skyblue", color = "black", trim = FALSE) +
  geom_boxplot(width = 0.1, fill = "white", outlier.size = 0.5) +
  theme_minimal() +
  labs(title = "Sample Size N = 2000, Number of Covariates P = 25", y = "Bias", x = "Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  +
  theme_minimal() + theme(plot.title = element_text(face = "italic", size = 12),
                          axis.title.x = element_text(face = "bold"),
                          axis.title.y = element_text(face = "bold"))

# Violin + Boxplot
ggplot(PEHE_long, aes(x = Model, y = PEHE)) +
  geom_violin(fill = "skyblue", color = "black", trim = FALSE) +
  geom_boxplot(width = 0.1, fill = "white", outlier.size = 0.5) +
  theme_minimal() +
  labs(title = "Sample Size N = 2000, Number of Covariates P = 25", y = "PEHE", x = "Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme_minimal() + theme(plot.title = element_text(face = "italic", size = 12),
                          axis.title.x = element_text(face = "bold"),
                          axis.title.y = element_text(face = "bold"))




