#### 07/13/2018 ####
library(glmnet)
library(MASS)
library(Rcpp)
sourceCpp("lassoALO.cpp")

# setup
n = 300
p = 80
k = 60
a = 0
true_beta = rnorm(p, 0, 1)
true_beta[-(1:k)] = 0

# misspecification example
X_mis = matrix(rnorm(n * p, 0, sqrt(1 / k)), n, p)
y_mis = X_mis %*% true_beta + rnorm(n, 0, 0.5)
y_mis[y_mis >= 0] = sqrt(y_mis[y_mis >= 0])
y_mis[y_mis < 0] = -sqrt(-y_mis[y_mis < 0])

# heavy-tailed noise example
X_hvy = matrix(rnorm(n * p, 0, sqrt(1 / k)), n, p)
t_noise = 0.5 * scale(rt(n, 3), center = F)
y_hvy = X_hvy %*% true_beta + t_noise

# correlated design example
C = toeplitz(0.8^(1:p))
X_cor = mvrnorm(n, rep(0, p), C / k)
y_cor = X_cor %*% true_beta + rnorm(n, 0, 0.5)

# search grid
# tune_param = 3.16 * 10^seq(-2, -3, length.out = 25)
# tune_param = 10^seq(-1.5, -2.5, length.out = 25)
tune_param = 10^seq(-5, -2, length.out = 25)

# LOOCV with glmnet
CV_mis = cv.glmnet(X_mis, y_mis, lambda = tune_param, nfolds = n, grouped = F, 
                   intercept = F, standardize = F, alpha = a)

CV_hvy = cv.glmnet(X_hvy, y_hvy, lambda = tune_param, nfolds = n, grouped = F, 
                   intercept = F, standardize = F, alpha = a)

CV_cor= cv.glmnet(X_cor, y_cor, lambda = tune_param, nfolds = n, grouped = F, 
                  intercept = F, standardize = F, alpha = a)

# ALOs
alo_mis = foreach(i = 1:length(CV_mis$glmnet.fit$lambda), .combine = c) %do% {
  alo_pred = elsnetALO(CV_mis$glmnet.fit$beta[, i], X_mis, y_mis, CV_mis$glmnet.fit$lambda[i], a)
  
  return(mean((alo_pred - y_mis)^2))
}
alo_hvy = foreach(i = 1:length(CV_hvy$glmnet.fit$lambda), .combine = c) %do% {
  alo_pred = elsnetALO(CV_hvy$glmnet.fit$beta[, i], X_hvy, y_hvy, CV_hvy$glmnet.fit$lambda[i], a)
  
  return(mean((alo_pred - y_hvy)^2))
}
alo_cor = foreach(i = 1:length(CV_cor$glmnet.fit$lambda), .combine = c) %do% {
  alo_pred = elsnetALO(CV_cor$glmnet.fit$beta[, i], X_cor, y_cor, CV_cor$glmnet.fit$lambda[i], a)
  
  return(mean((alo_pred - y_cor)^2))
}

# plots
par(mfrow = c(1, 3))
plot(rev(CV_mis$cvm), xlab = "lambda", ylab = "Elastic Net risk, a = 0",type = "l", lwd = 2, col = "darkorange")
lines(rev(alo_mis), type = "b", pch = 4, lwd = 2, col = 4)

plot(rev(CV_hvy$cvm), xlab = "lambda", ylab = "Elastic Net risk, a = 0",type = "l", lwd = 2, col = "darkorange")
lines(rev(alo_hvy), type = "b", pch = 4, lwd = 2, col = 4)

plot(rev(CV_cor$cvm), xlab = "lambda", ylab = "Elastic Net risk, a = 0",type = "l", lwd = 2, col = "darkorange")
lines(rev(alo_cor), type = "b", pch = 4, lwd = 2, col = 4)
