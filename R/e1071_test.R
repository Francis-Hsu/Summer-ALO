#### 09/11/2018 ####
library(Rcpp)
library(e1071)
library(foreach)
library(doParallel)
sourceCpp("svmALO.cpp")

#### Data ####
n = 300
p = 50
m = 25

X = matrix(rnorm(n * p, 0, 1), n, p)
X = scale(X)
beta = rnorm(p, 0, 3)
y = rbinom(n, size = 1, prob = 1 / (1 + exp(-X %*% beta)))
y[y == 0] = -1
y.fac = as.factor(y)
lead_sign = sign(y[1]) # LIBSVM treats the first y as +1, no matter the actual value

#### Tuning Parameters ####
lambda = exp(seq(-2, 6, length.out = m))

#### RBF ####
g = 2 / p
K = gaussianKer(X, g)

### LOO ###
cl = makeCluster(4)
registerDoParallel(cl)
LOO_pred = foreach(i = 1:m, .combine = 'cbind', .packages = "e1071") %:% foreach(j = 1:n, .combine = 'c') %dopar% {
  fit = svm(X[-j, ], y.fac[-j], scale = FALSE, kernel = "radial", 
            cost = 1 / lambda[i], gamma = g, tolerance = 1e-6) # default tol = 0.001
  alpha = rep(0, n - 1)
  alpha[fit$index] = lead_sign * fit$coefs
  rho = -lead_sign * fit$rho
  y_hat = K[-j, j] %*% alpha + rho
  
  return(y_hat)
}
stopCluster(cl)

### ALO Method 1 ###
ALO_pred = foreach(i = 1:m, .combine = 'cbind') %do% {
  fit = svm(X, y.fac, scale = FALSE, kernel = "radial", 
            cost = 1 / lambda[i], gamma = g, tolerance = 1e-8, epsilon = 0.0001)
  alpha = rep(0, n)
  alpha[fit$index] = lead_sign * fit$coefs
  rho = -lead_sign * fit$rho
  y_hat = svmKerALO(K, y, alpha, rho, lambda[i], 1e-5) 
  
  return(y_hat)
}

# plots
tikz('rbf_2_1.tex', width = 6, height = 3)
par(mfrow = c(1, 2), mar = c(4, 4, 1, 2) + 0.1)

# Class error
LOO_risk = colMeans(y != sign(LOO_pred))
ALO_risk = colMeans(y != sign(ALO_pred))
plot(LOO_risk, type = "l", lwd = 2, ylim = range(LOO_risk, ALO_risk), col = "darkorange", ylab = "Class")
lines(ALO_risk, type = "b", pch = 4, lwd = 2, col = 4)
legend("bottomright", legend = c("LOO", "ALO"), col = c(2, 4), pch = c(NA, 4), lwd = 2, bty = "n")

# MSE
LOO_risk = sapply(1:m, function(i) { mean((y - LOO_pred[, i])^2) })
ALO_risk = sapply(1:m, function(i) { mean((y - ALO_pred[, i])^2) })
plot(LOO_risk, type = "l", lwd = 2, col = "darkorange", ylab = "MSE")
lines(ALO_risk, type = "b", pch = 4, lwd = 2, col = 4)
legend("bottomright", legend = c("LOO", "ALO"), col = c(2, 4), pch = c(NA, 4), lwd = 2, bty = "n")
dev.off()

### ALO Method 2 ###
X_feat = gaussianKerApprox(X, g)
ALO_pred = foreach(i = 1:m, .combine = 'cbind') %do% {
  svm_lin = svm(X_feat, y.fac, scale = FALSE, kernel = "linear", 
                cost = 1 / lambda[i], gamma = g, tolerance = 1e-6)
  W = lead_sign * t(svm_lin$coefs) %*% svm_lin$SV
  b = -lead_sign * svm_lin$rho
  y_hat = svmALO(X_feat, y, W, b, lambda[i], 1e-5)
  
  return(y_hat)
}

# plots
tikz('rbf_2_2.tex', width = 6, height = 3)
par(mfrow = c(1, 2), mar = c(4, 4, 1, 2) + 0.1)

# Class error
LOO_risk = colMeans(y != sign(LOO_pred))
ALO_risk = colMeans(y != sign(ALO_pred))
plot(LOO_risk, type = "l", lwd = 2, ylim = range(LOO_risk, ALO_risk), col = "darkorange", ylab = "Class")
lines(ALO_risk, type = "b", pch = 4, lwd = 2, col = 4)
legend("bottomright", legend = c("LOO", "ALO"), col = c(2, 4), pch = c(NA, 4), lwd = 2, bty = "n")

# MSE
LOO_risk = sapply(1:m, function(i) { mean((y - LOO_pred[, i])^2) })
ALO_risk = sapply(1:m, function(i) { mean((y - ALO_pred[, i])^2) })
plot(LOO_risk, type = "l", lwd = 2, col = "darkorange", ylab = "MSE")
lines(ALO_risk, type = "b", pch = 4, lwd = 2, col = 4)
legend("bottomright", legend = c("LOO", "ALO"), col = c(2, 4), pch = c(NA, 4), lwd = 2, bty = "n")
dev.off()

#### Polynomial ####
g = 2 / p
d = 5
c0 = 0.25
K = polynomialKer(X, g, c0, d)

### LOO ###
cl = makeCluster(4)
registerDoParallel(cl)
LOO_pred = foreach(i = 1:m, .combine = 'cbind', .packages = "e1071") %:% foreach(j = 1:n, .combine = 'c') %dopar% {
  fit = svm(X[-j, ], y.fac[-j], scale = FALSE, kernel = "polynomial", 
            cost = 1 / lambda[i], gamma = g, coef0 = c0, degree = d, tolerance = 1e-6) # default tol = 0.001
  alpha = rep(0, n - 1)
  alpha[fit$index] = lead_sign * fit$coefs
  rho = -lead_sign * fit$rho
  y_hat = K[-j, j] %*% alpha + rho
  
  return(y_hat)
}
stopCluster(cl)

### ALO Method 1 ###
ALO_pred = foreach(i = 1:m, .combine = 'cbind') %do% {
  fit = svm(X, y.fac, scale = FALSE, kernel = "polynomial", 
            cost = 1 / lambda[i], gamma = g, coef0 = c0, degree = d, tolerance = 1e-8, epsilon = 0.0001)
  alpha = rep(0, n)
  alpha[fit$index] = lead_sign * fit$coefs
  rho = -lead_sign * fit$rho
  y_hat = svmKerALO(K, y, alpha, rho, lambda[i], 1e-5) 
  
  return(y_hat)
}

# plots
tikz('pol_2_1.tex', width = 6, height = 3)
par(mfrow = c(1, 2), mar = c(4, 4, 1, 2) + 0.1)

# Class error
LOO_risk = colMeans(y != sign(LOO_pred))
ALO_risk = colMeans(y != sign(ALO_pred))
plot(LOO_risk, type = "l", lwd = 2, ylim = range(LOO_risk, ALO_risk), col = "darkorange", ylab = "Class")
lines(ALO_risk, type = "b", pch = 4, lwd = 2, col = 4)
legend("topleft", legend = c("LOO", "ALO"), col = c(2, 4), pch = c(NA, 4), lwd = 2, bty = "n")

# MSE
LOO_risk = sapply(1:m, function(i) { mean((y - LOO_pred[, i])^2) })
ALO_risk = sapply(1:m, function(i) { mean((y - ALO_pred[, i])^2) })
plot(LOO_risk, type = "l", lwd = 2, col = "darkorange", ylab = "MSE")
lines(ALO_risk, type = "b", pch = 4, lwd = 2, col = 4)
legend("topleft", legend = c("LOO", "ALO"), col = c(2, 4), pch = c(NA, 4), lwd = 2, bty = "n")
dev.off()

### ALO Method 2 ###
X_feat = polynomialKerApprox(X, g, c0, d)
ALO_pred = foreach(i = 1:m, .combine = 'cbind') %do% {
  svm_lin = svm(X_feat, y.fac, scale = FALSE, kernel = "linear", 
                cost = 1 / lambda[i], gamma = g, tolerance = 1e-6)
  W = lead_sign * t(svm_lin$coefs) %*% svm_lin$SV
  b = -lead_sign * svm_lin$rho
  y_hat = svmALO(X_feat, y, W, b, lambda[i], 1e-5)
  
  return(y_hat)
}

# plots
tikz('pol_2_2.tex', width = 6, height = 3)
par(mfrow = c(1, 2), mar = c(4, 4, 1, 2) + 0.1)

# Class error
LOO_risk = colMeans(y != sign(LOO_pred))
ALO_risk = colMeans(y != sign(ALO_pred))
plot(LOO_risk, type = "l", lwd = 2, ylim = range(LOO_risk, ALO_risk), col = "darkorange", ylab = "Class")
lines(ALO_risk, type = "b", pch = 4, lwd = 2, col = 4)
legend("topleft", legend = c("LOO", "ALO"), col = c(2, 4), pch = c(NA, 4), lwd = 2, bty = "n")

# MSE
LOO_risk = sapply(1:m, function(i) { mean((y - LOO_pred[, i])^2) })
ALO_risk = sapply(1:m, function(i) { mean((y - ALO_pred[, i])^2) })
plot(LOO_risk, type = "l", lwd = 2, col = "darkorange", ylab = "MSE")
lines(ALO_risk, type = "b", pch = 4, lwd = 2, col = 4)
legend("topleft", legend = c("LOO", "ALO"), col = c(2, 4), pch = c(NA, 4), lwd = 2, bty = "n")
dev.off()

#### Sigmoid ####
g = 2 / p
c0 = 0.3
K = sigmoidKer(X, g, c0)

### LOO ###
cl = makeCluster(4)
registerDoParallel(cl)
LOO_pred = foreach(i = 1:m, .combine = 'cbind', .packages = "e1071") %:% foreach(j = 1:n, .combine = 'c') %dopar% {
  fit = svm(X[-j, ], y.fac[-j], scale = FALSE, kernel = "sigmoid", 
            cost = 1 / lambda[i], gamma = g, coef0 = c0, tolerance = 1e-6) # default tol = 0.001
  alpha = rep(0, n - 1)
  alpha[fit$index] = lead_sign * fit$coefs
  rho = -lead_sign * fit$rho
  y_hat = K[-j, j] %*% alpha + rho
  
  return(y_hat)
}
stopCluster(cl)

### ALO Method 1 ###
ALO_pred = foreach(i = 1:m, .combine = 'cbind') %do% {
  fit = svm(X, y.fac, scale = FALSE, kernel = "sigmoid", 
            cost = 1 / lambda[i], gamma = g, coef0 = c0, tolerance = 1e-6)
  alpha = rep(0, n)
  alpha[fit$index] = lead_sign * fit$coefs
  rho = -lead_sign * fit$rho
  y_hat = svmKerALO(K, y, alpha, rho, lambda[i], 1e-5) 
  
  return(y_hat)
}

# plots
tikz('sig_1.tex', width = 6, height = 3)
par(mfrow = c(1, 2), mar = c(4, 4, 1, 2) + 0.1)

# Class error
LOO_risk = colMeans(y != sign(LOO_pred))
ALO_risk = colMeans(y != sign(ALO_pred))
plot(LOO_risk, type = "l", lwd = 2, ylim = range(LOO_risk, ALO_risk), col = "darkorange", ylab = "Class")
lines(ALO_risk, type = "b", pch = 4, lwd = 2, col = 4)
legend("topleft", legend = c("LOO", "ALO"), col = c(2, 4), pch = c(NA, 4), lwd = 2, bty = "n")

# MSE
LOO_risk = sapply(1:m, function(i) { mean((y - LOO_pred[, i])^2) })
ALO_risk = sapply(1:m, function(i) { mean((y - ALO_pred[, i])^2) })
plot(LOO_risk, type = "l", lwd = 2, col = "darkorange", ylab = "MSE")
lines(ALO_risk, type = "b", pch = 4, lwd = 2, col = 4)
legend("topright", legend = c("LOO", "ALO"), col = c(2, 4), pch = c(NA, 4), lwd = 2, bty = "n")
dev.off()

# visual check
plot(ALO_pred[, 1])
points(LOO_pred[, 1], col = 2)

plot(ALO_pred[, 15])
points(LOO_pred[, 15], col = 2)

### ALO Method 2 ###
X_feat = sigmoidKerApprox(X, g, c0)
ALO_pred = foreach(i = 1:m, .combine = 'cbind') %do% {
  svm_lin = svm(X_feat, y.fac, scale = FALSE, kernel = "linear", 
                cost = 1 / lambda[i], gamma = g, tolerance = 1e-6)
  W = lead_sign * t(svm_lin$coefs) %*% svm_lin$SV
  b = -lead_sign * svm_lin$rho
  y_hat = svmALO(X_feat, y, W, b, lambda[i], 1e-5)
  
  return(y_hat)
}

# plots
tikz('sig_2.tex', width = 6, height = 3)
par(mfrow = c(1, 2), mar = c(4, 4, 1, 2) + 0.1)

# Class error
LOO_risk = colMeans(y != sign(LOO_pred))
ALO_risk = colMeans(y != sign(ALO_pred))
plot(LOO_risk, type = "l", lwd = 2, ylim = range(LOO_risk, ALO_risk), col = "darkorange", ylab = "Class")
lines(ALO_risk, type = "b", pch = 4, lwd = 2, col = 4)
legend("bottomright", legend = c("LOO", "ALO"), col = c(2, 4), pch = c(NA, 4), lwd = 2, bty = "n")

# MSE
LOO_risk = sapply(1:m, function(i) { mean((y - LOO_pred[, i])^2) })
ALO_risk = sapply(1:m, function(i) { mean((y - ALO_pred[, i])^2) })
plot(LOO_risk, type = "l", lwd = 2, col = "darkorange", ylab = "MSE")
lines(ALO_risk, type = "b", pch = 4, lwd = 2, col = 4)
legend("bottomright", legend = c("LOO", "ALO"), col = c(2, 4), pch = c(NA, 4), lwd = 2, bty = "n")
dev.off()

# visual check
plot(ALO_pred[, 1])
points(LOO_pred[, 1], col = 2)

plot(ALO_pred[, 15])
points(LOO_pred[, 15], col = 2)
