library(glmnet)
library(rbenchmark)
library(Rcpp)
sourceCpp("lassoALO.cpp")
sourceCpp("matUpdate.cpp")
source("aloWrappers.R")

#########
# setup
n = 500
p = 200
k = 20
true_beta = rnorm(p, 0, 1)
true_beta[-(1:k)] = 0

# misspecification example
X = matrix(rnorm(n * p, 0, sqrt(1 / k)), n, p)
y = X %*% true_beta + rnorm(n, 0, 0.5)
y[y >= 0] = sqrt(y[y >= 0])
y[y < 0] = -sqrt(-y[y < 0])
sd = c(sd(y) * sqrt(n - 1) / sqrt(n))
y = y / sd

#tune_param = 10^seq(-3, -1.5, length.out = 25)
fit = glmnet(X, y, nlambda = 25, standardize = F, intercept = F)

ptm = proc.time()
#mse0 = cv.glmnet(X, y, nfolds = n, nlambda = 25, standardize = F, grouped = F)$cvm
proc.time() - ptm

ptm = proc.time()
mse1 = lassoALO.Vanilla(X, y, fit)
proc.time() - ptm

ptm = proc.time()
mse2 = lassoALO.Woodbury(X, y, fit)
proc.time() - ptm

plot(mse1, type = "l", col = "orange", lwd = 2)
# lines(mse2, type = "b", col = 6, lwd = 2)
lines(mse2, type = "b", col = 4, pch = 3, lwd = 2)

benchmark(lassoALO_vanilla(X, y, fit), lassoALO_woodbury(X, y, fit), replications = 50)

###########################
a = 0.5
fit = glmnet(X, y, nlambda = 25, standardize = F, intercept = F, alpha = a)
fit = cv.glmnet(X, y, nfolds = n, nlambda = 25, standardize = F, grouped = F, intercept = F, alpha = a)
mse0 = fit$cvm

ptm = proc.time()
mse1 = elnetALO.Vanilla(X, y, a, fit$glmnet.fit)
proc.time() - ptm

ptm = proc.time()
mse2 = elnetALO.Approx(X, y, a, fit$glmnet.fit)
proc.time() - ptm

plot(mse0, type = "l", col = "orange", lwd = 2)
lines(mse1, type = "b", col = 6, lwd = 2)
lines(mse2, type = "b", col = 4, pch = 3, lwd = 2)
