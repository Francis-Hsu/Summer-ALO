devtools::install_github("Francis-Hsu/alocv")
library(alocv)
library(doParallel)
library(MASS)

# compare result from cv.glmnet and glmnetALO
# note that cv.glmnet will usually skip some lambdas
# so refit a model is necessary for comparison

######## Linear ########
# data
n = 300
p = 100
k = 60
true_beta = rnorm(p, 0, 1)
true_beta[-(1:k)] = 0

X = matrix(rnorm(n * p, 0, sqrt(1 / k)), n, p)
y = X %*% true_beta + rnorm(n, 0, 0.5)
y[y >= 0] = 2 * sqrt(y[y >= 0])
y[y < 0] = -2 * sqrt(-y[y < 0])

# params
is_intr = T
is_stdz = T
measure = "mse"

par(mfrow = c(1, 3))
for(a in c(0, 0.5, 1)) {
  CV_el = cv.glmnet(X, y, alpha = a, nfolds = n, grouped = F, 
                    intercept = is_intr, standardize = is_stdz, type.measure = measure)
  glm_obj_CV = glmnet(X, y, lambda = CV_el$lambda, alpha = a, 
                      intercept = is_intr, standardize = is_stdz)
  ALO_el = glmnetALO(X, y, glm_obj_CV, a, standardize = is_stdz, type.measure = measure)
  
  # old school
  plot(rev(CV_el$cvm), xlab = "lambda", ylab = "Risk", type = "l", lwd = 2, col = "darkorange")
  lines(rev(ALO_el$alom), type = "b", pch = 4, lwd = 2, col = 4)
}
par(mfrow = c(2, 1))
plot(CV_el)
plot(ALO_el)

######## Logistic ########
# data
n = 300
p = 100
k = 60
true_beta = rnorm(p, 0, 4)
true_beta[-(1:k)] = 0

X = matrix(rnorm(n * p, 0, sqrt(1 / k)), n, p)
y = rbinom(n, size = 1, prob = 1 / (1 + exp(-X %*% true_beta)))

# params
is_intr = T
is_stdz = T
measure = "mse"

par(mfrow = c(1, 3))
for(a in c(0, 0.5, 1)) {
  CV_bin = cv.glmnet(X, y, family = "binomial", alpha = a, nlambda = 35, nfolds = n, grouped = F, 
                     intercept = is_intr, standardize = is_stdz, type.measure = measure)
  glm_obj_CV = glmnet(X, y, family = "binomial", lambda = CV_bin$lambda, alpha = a, 
                      intercept = is_intr, standardize = is_stdz)
  ALO_bin = glmnetALO(X, y, glm_obj_CV, a, standardize = is_stdz, type.measure = measure)
  
  # old school
  plot(rev(CV_bin$cvm), xlab = "lambda", ylab = "Risk", type = "l", lwd = 2, col = "darkorange")
  lines(rev(ALO_bin$alom), type = "b", pch = 4, lwd = 2, col = 4)
}
par(mfrow = c(2, 1))
plot(CV_bin)
plot(ALO_bin)

######## Poisson ########
# data
n = 300
p = 100
k = 60
true_beta = rnorm(p, 0, 0.25)
true_beta[-(1:k)] = 0

X = matrix(rnorm(n * p, 1 / k, sqrt(1 / k)), n, p)
y = rpois(n, lambda = exp(1 + X %*% true_beta))

# params
is_intr = T
is_stdz = F
measure = "mse"

par(mfrow = c(1, 3))
for(a in c(0, 0.5, 1)) {
  CV_pois = cv.glmnet(X, y, family = "poisson", alpha = a, nlambda = 25, nfolds = n, grouped = F, 
                      intercept = is_intr, standardize = is_stdz, type.measure = measure)
  glm_obj_CV = glmnet(X, y, family = "poisson", lambda = CV_pois$lambda, alpha = a, 
                      intercept = is_intr, standardize = is_stdz)
  ALO_pois = glmnetALO(X, y, glm_obj_CV, a, standardize = is_stdz, type.measure = measure)
  
  # old school
  plot(rev(CV_pois$cvm), xlab = "lambda", ylab = "Risk", type = "l", lwd = 2, col = "darkorange")
  lines(rev(ALO_pois$alom), type = "b", pch = 4, lwd = 2, col = 4)
}
par(mfrow = c(2, 1))
plot(CV_pois)
plot(ALO_pois)

######## Multinomial ########
# data
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

# params
lambda = 10^seq(-3.5, -0.5, length.out = 25)
is_intr = T
is_stdz = F
measure = "mse"

# TODO: cv.glmnet has convergence issue with some lambda choice
# TODO: standardization may still have some issues, need investigate
cl = makeCluster(detectCores())
registerDoParallel(cl)
par(mfrow = c(1, 3))
for(a in c(0, 0.5, 1)) {
  CV_ml = cv.glmnet(X, y.mat, family = "multinomial", alpha = a, lambda = lambda, nfolds = n, grouped = F, 
                    intercept = is_intr, standardize = is_stdz, type.measure = measure, parallel = T, maxit = 1e06)
  glm_obj_CV = glmnet(X, y.mat, family = "multinomial", lambda = CV_ml$lambda, alpha = a, 
                      intercept = is_intr, standardize = is_stdz)
  ALO_ml = glmnetALO(X, y.mat, glm_obj_CV, a, standardize = is_stdz, type.measure = measure)
  
  # old school
  plot(rev(CV_ml$cvm), xlab = "lambda", ylab = "Risk", type = "l", lwd = 2, col = "darkorange")
  lines(rev(ALO_ml$alom), type = "b", pch = 4, lwd = 2, col = 4)
}
stopCluster(cl)
par(mfrow = c(2, 1))
plot(CV_ml)
plot(ALO_ml)
