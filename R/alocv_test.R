library(glmnet)
library(MASS)
library(Rcpp)
sourceCpp("glmnetALO.cpp")
source("glmnetALO.R")
source("plot.glmnetALO.R")
source("glmnetALO.risk.R")
source("glmnetALO.unscale.coef.R")

######## Elastic Net ########
n = 300
p = 100
k = 60
true_beta = rnorm(p, 0, 1)
true_beta[-(1:k)] = 0

X = matrix(rnorm(n * p, 0, sqrt(1 / k)), n, p)
y = X %*% true_beta + rnorm(n, 0, 0.5)
y[y >= 0] = 2 * sqrt(y[y >= 0])
y[y < 0] = -2 * sqrt(-y[y < 0])

par(mfrow = c(1, 3))
for(a in c(0, 0.5, 1)) {
  CV_el = cv.glmnet(X, y, nfolds = n, grouped = F, intercept = T, standardize = T, alpha = a, type.measure = "mse")
  ALO_el = glmnetALO(X, y, a, CV_el$glmnet.fit, type.measure = "mse", standardize = T)
  
  # old school
  plot(rev(CV_el$cvm), xlab = "lambda", ylab = "Risk", type = "l", lwd = 2, col = "darkorange")
  lines(rev(ALO_el$riskm), type = "b", pch = 4, lwd = 2, col = 4)
}
par(mfrow = c(1, 1))
plot(CV_el)
plot(ALO_el)

######## Logistic ########
n = 300
p = 100
k = 60
true_beta = rnorm(p, 0, 4)
true_beta[-(1:k)] = 0

X = matrix(rnorm(n * p, 0, sqrt(1 / k)), n, p)
y = rbinom(n, size = 1, prob = 1 / (1 + exp(-X %*% true_beta)))

# TODO: index bug?
par(mfrow = c(1, 3))
for(a in c(0, 0.5, 1)) {
  CV_bin = cv.glmnet(X, y, family = "binomial", nlambda = 50, nfolds = n, grouped = F, 
                     intercept = T, standardize = F, alpha = a, type.measure = "mse")
  ALO_bin = glmnetALO(X, y, a, CV_bin$glmnet.fit, type.measure = "mse", standardize = F)
  
  # old school
  plot(rev(CV_bin$cvm), xlab = "lambda", ylab = "Risk", type = "l", lwd = 2, col = "darkorange")
  lines(rev(ALO_bin$riskm), type = "b", pch = 4, lwd = 2, col = 4)
}

######## Poisson ########
n = 300
p = 100
k = 60
true_beta = abs(rnorm(p, 0, 0.25))
true_beta[-(1:k)] = 0

X = abs(matrix(rnorm(n * p, 1 / k, sqrt(1 / k)), n, p))
y = rpois(n, lambda = exp(X %*% true_beta)) + 1

# TODO: standardization not working
par(mfrow = c(1, 3))
for(a in c(0, 0.5, 1)) {
  CV_pois = cv.glmnet(X, y, family = "poisson", nlambda = 25, nfolds = n, grouped = F, 
                      intercept = T, standardize = F, alpha = a, type.measure = "mse")
  
  ALO_pois = glmnetALO(X, y, a, CV_pois$glmnet.fit, type.measure = "mse", standardize = F)
  
  # old school
  plot(rev(CV_pois$cvm), xlab = "lambda", ylab = "Risk", type = "l", lwd = 2, col = "darkorange")
  lines(rev(ALO_pois$riskm), type = "b", pch = 4, lwd = 2, col = 4)
}
plot(CV_pois)
plot(ALO_pois)

######## Multinomial ########
n = 250
p = 100
k = 60
num_class = 3

beta = matrix(rnorm(num_class * p, mean = 0, sd = 1), ncol = num_class)
beta[(k + 1):p,] = 0
intercept = rnorm(num_class, mean = 0, sd = 1)
X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
y.linear = matrix(rep(intercept, n), ncol = num_class, byrow = TRUE) +
  X %*% beta
prob = exp(y.linear) / matrix(rep(rowSums(exp(y.linear)), num_class), ncol =
                                num_class)
y.mat = t(apply(prob, 1, function(x) rmultinom(1, 1, prob = x))) # N * K matrix (N - #obs, K - #class)

# TODO: issue with some lambda choice
par(mfrow = c(1, 3))
for(a in c(0, 0.5, 1)) {
  CV_ml = cv.glmnet(X, y.mat, family = "multinomial", nlambda = 25, nfolds = n, grouped = F, 
                    intercept = T, standardize = F, alpha = a, type.measure = "mse")
  ALO_ml = glmnetALO(X, y.mat, a, CV_ml$glmnet.fit, standardize = F)
  
  # old school
  plot(rev(CV_ml$cvm), xlab = "lambda", ylab = "Risk", type = "l", lwd = 2, col = "darkorange")
  lines(rev(ALO_ml$riskm), type = "b", pch = 4, lwd = 2, col = 4)
}
plot(CV_ml)
plot(ALO_ml)
