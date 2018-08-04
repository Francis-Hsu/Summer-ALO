library(glmnet)
library(rbenchmark)
library(Rcpp)
sourceCpp("lassoALO.cpp")
sourceCpp("cholInv.cpp")

#########
lassoALO_chol = function(X, y, glm_obj, tune_param) {
  # rip stuffs from the glmnet object
  glm_lambda = glm_obj$lambda # assume at least 2 lambdas
  glm_beta = glm_obj$beta
  M = length(glm_lambda)
  
  # find update/downdate indices
  glm_active_idx = apply(glm_beta, 2, function(b) { unname(which(abs(b) >= 1e-8)) })
  glm_add_idx = lapply(2:M, function(i) {
    setdiff(glm_active_idx[[i]], glm_active_idx[[i - 1]])
  })
  glm_rmv_idx = lapply(2:M, function(i) {
    setdiff(glm_active_idx[[i - 1]], glm_active_idx[[i]])
  })

  alo_update = lassoALOChol(X, y, as.matrix(glm_beta), glm_active_idx, glm_add_idx, glm_rmv_idx)
  mse = colMeans(alo_update^2)
  
  return(mse)
}

lassoALO_vanilla = function(X, y, glm_obj) {
  glm_lambda = glm_obj$lambda # assume at least 2 lambdas
  glm_beta = glm_obj$beta
  
  Y_alo = foreach(i = 1:length(glm_lambda), .combine = cbind) %do% {
    temp_lambda = glm_lambda[i]
    temp_beta = glm_beta[, i]
    alo_pred = elnetALO(temp_beta, X, y, temp_lambda, 1)
  }
  
  return(colMeans(sweep(Y_alo, 1, y)^2))
}

# setup
n = 4000
p = 4000
k = 600
true_beta = rnorm(p, 0, 1)
true_beta[-(1:k)] = 0

# misspecification example
X = matrix(rnorm(n * p, 0, sqrt(1 / k)), n, p)
y = X %*% true_beta + rnorm(n, 0, 0.5)
y[y >= 0] = sqrt(y[y >= 0])
y[y < 0] = -sqrt(-y[y < 0])
sd = c(sd(y) * sqrt(n - 1) / sqrt(n))
y = y / sd

tune_param = 10^seq(-3, -1.5, length.out = 25)
fit = glmnet(X, y, lambda = tune_param)

ptm = proc.time()
mse1 = lassoALO_vanilla(X, y, fit)
proc.time() - ptm

ptm = proc.time()
mse2 = lassoALO_chol(X, y, fit)
proc.time() - ptm

plot(mse1, type = "l", col = "orange")
lines(mse2, type = "b")

benchmark(lassoALO_vanilla(X, y, fit), lassoALO_chol(X, y, fit), replications = 10)