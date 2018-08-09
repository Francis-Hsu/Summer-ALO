# Elastic Net -------------------------------------------------------------
setwd("E:\\Columbia_University\\Internship\\R_File\\LASSO\\")
library(glmnet)
library(ggplot2)
library(Rcpp)
sourceCpp("src/ALO_Primal.cpp")
source("R/ElasticNet_Functions.R")


# Elastic Net with Intercept ----------------------------------------------

# misspecification --------------------------------------------------------

# parameters
n = 1000
p = 1500
k = 500
set.seed(1234)

# simulation
beta = rnorm(p, mean = 0, sd = 1)
beta[(k + 1):p] = 0
intercept = 1
X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
sigma = rnorm(n, mean = 0, sd = 0.5)
y = intercept + X %*% beta + sigma
index = which(y >= 0)
y[index] = sqrt(y[index])
y[-index] = -sqrt(-y[-index])

# find lambda
sd.y = as.numeric(sqrt(var(y) * length(y) / (length(y) - 1)))
y.scaled = y / sd.y
X.scaled = X / sd.y
model = glmnet(
  x = X.scaled,
  y = y.scaled,
  family = "gaussian",
  alpha = 1,
  # thresh = 1E-14,
  intercept = TRUE,
  standardize = FALSE,
  maxit = 1000000,
  nlambda = 20
)
lambda = sort(model$lambda * sd.y ^ 2, decreasing = TRUE)
alpha = 1
param = data.frame(alpha = numeric(0),
                   lambda = numeric(0))
for (i in 1:length(alpha)) {
  for (j in 1:length(lambda)) {
    param[j + (i - 1) * length(lambda), c('alpha', 'lambda')] = c(alpha[i], lambda[j])
  }
}

# normal ALO function
ElasticNet_ALO = function(X, y, param, alpha, lambda) {
  # compute the scale parameter for y
  sd.y = as.numeric(sqrt(var(y) * length(y) / (length(y) - 1)))
  y.scaled = y / sd.y
  X.scaled = X / sd.y
  # find the ALO prediction
  y.alo = matrix(ncol = dim(param)[1], nrow = n)
  for (k in 1:length(alpha)) {
    # build the full data model
    model = glmnet(
      x = X.scaled,
      y = y.scaled,
      family = "gaussian",
      alpha = alpha[k],
      lambda = lambda / sd.y ^ 2,
      # thresh = 1E-14,
      intercept = TRUE,
      standardize = FALSE,
      maxit = 1000000
    )
    # find the prediction for each alpha value
    y.temp <- foreach(j = 1:length(lambda), .combine = cbind) %do% {
      ElasticNetALO(as.vector(model$beta[, j]),
                    model$a0[j] * sd.y,
                    X,
                    y,
                    lambda[j],
                    alpha[k])
    }
    y.alo[, ((k - 1) * length(lambda) + 1):(k * length(lambda))] = y.temp
  }
  # true leave-one-out risk estimate
  risk.alo = 1 / n * colSums((y.alo -
                                matrix(rep(y, dim(
                                  param
                                )[1]), ncol = dim(param)[1])) ^ 2)
  
  # return risk estimate
  return(risk.alo)
}


# Cholesky decomposition ALO
ElasticNet_ALO_Chol = function(X, y, param, alpha, lambda) {
  # compute the scale parameter for y
  sd.y = as.numeric(sqrt(var(y) * length(y) / (length(y) - 1)))
  y.scaled = y / sd.y
  X.scaled = X / sd.y
  # find the ALO prediction
  y.alo = matrix(ncol = dim(param)[1], nrow = n)
  for (k in 1:length(alpha)) {
    # build the full data model
    model = glmnet(
      x = X.scaled,
      y = y.scaled,
      family = "gaussian",
      alpha = alpha[k],
      lambda = lambda / sd.y ^ 2,
      # thresh = 1E-14,
      intercept = TRUE,
      standardize = FALSE,
      maxit = 1000000
    )
    # re-order variables
    beta.hat = as.matrix(model$beta)
    X.order = X
    X.scaled.order = X.scaled
    for (j in 2:ncol(beta.hat)) {
      idx.front = which((beta.hat[, j] != 0 &
                           beta.hat[, j - 1] != 0) |
                          beta.hat[, j - 1] == 0)
      idx.end = which(beta.hat[, j] == 0 & beta.hat[, j - 1] != 0)
      idx = c(idx.front, idx.end)
      beta.hat = beta.hat[idx,]
      X.order = X.order[, idx]
      X.scaled.ordered = X.scaled.order[, idx]
    }
    XtX.order = t(cbind(1, X.order)) %*% cbind(1, X.order)
    # find the prediction for each alpha value
    if (alpha[k] == 1) {
      L = matrix(ncol = 0, nrow = 0)
      idx_old = numeric(0)
      for (j in 1:length(lambda)) {
        cat(j, 'th out of ', length(lambda), '\n', sep = '')
        update = ElasticNetALO_CholUpdate(
          as.vector(beta.hat[, j]),
          model$a0[j] * sd.y,
          X.order,
          y,
          lambda[j],
          alpha[k],
          L,
          idx_old,
          XtX.order
        )
        y.alo[, (k - 1) * length(lambda) + j] = update[[1]]
        L = update[[2]]
        idx_old = as.vector(update[[3]])
      }
    } else {
      y.temp <- foreach(j = 1:length(lambda), .combine = cbind) %do% {
        ElasticNetALO(as.vector(model$beta[, j]),
                      model$a0[j] * sd.y,
                      X,
                      y,
                      lambda[j],
                      alpha[k])
      }
      y.alo[, ((k - 1) * length(lambda) + 1):(k * length(lambda))] = y.temp
    }
  }
  # true leave-one-out risk estimate
  risk.alo = 1 / n * colSums((y.alo -
                                matrix(rep(y, dim(
                                  param
                                )[1]), ncol = dim(param)[1])) ^ 2)
  # return risk estimate
  return(risk.alo)
}

# block inversiong update
ElasticNet_ALO_Block = function(X, y, param, alpha, lambda) {
  # compute the scale parameter for y
  sd.y = as.numeric(sqrt(var(y) * length(y) / (length(y) - 1)))
  y.scaled = y / sd.y
  X.scaled = X / sd.y
  # find the ALO prediction
  y.alo = matrix(ncol = dim(param)[1], nrow = n)
  for (k in 1:length(alpha)) {
    # build the full data model
    model = glmnet(
      x = X.scaled,
      y = y.scaled,
      family = "gaussian",
      alpha = alpha[k],
      lambda = lambda / sd.y ^ 2,
      # thresh = 1E-14,
      intercept = TRUE,
      standardize = FALSE,
      maxit = 1000000
    )
    # compute some variables
    beta.hat = matrix(ncol = length(lambda), nrow = p + 1)
    beta.hat[1,] = model$a0 * sd.y
    beta.hat[2:(p + 1),] = as.matrix(model$beta)
    X.full = cbind(1, X)
    XtX = t(X.full) %*% X.full
    # find the prediction for each alpha value
    if (alpha[k] == 1) {
      A.inv = matrix(ncol = 0, nrow = 0)
      E.old = numeric(0)
      for (j in 1:length(lambda)) {
        cat(j, 'th out of ', length(lambda), '\n', sep = '')
        # find the active set
        E = which(beta.hat[, j] != 0)
        # drop rows & cols
        n_drop = sum(E.old %in% E == FALSE)
        if (n_drop > 0) {
          keep.pos = which(E.old %in% E)
          drop.pos = which(E.old %in% E == FALSE)
          A.inv = A.inv[c(keep.pos, drop.pos), c(keep.pos, drop.pos)]
          A.inv = BlockInverse_Drop(A.inv, length(keep.pos))
          E.old = E.old[keep.pos]
        }
        # add rows & cols
        n_add = sum(E %in% E.old == FALSE)
        if (n_add > 0) {
          E.add = E[which(E %in% E.old == FALSE)]
          A.inv = BlockInverse_Add(XtX, E.old - 1, E.add - 1, A.inv)
          E.old = c(E.old, E.add)
          idx = order(E.old, decreasing = FALSE)
          A.inv = as.matrix(A.inv[idx, idx])
          E.old = E.old[idx]
        }
        y.alo[, (k - 1) * length(lambda) + j] =
          BlockInverse_ALO(X.full, A.inv, y, beta.hat[, j], E.old - 1)
      }
    } else {
      y.temp <- foreach(j = 1:length(lambda), .combine = cbind) %do% {
        ElasticNetALO(as.vector(model$beta[, j]),
                      model$a0[j] * sd.y,
                      X,
                      y,
                      lambda[j],
                      alpha[k])
      }
      y.alo[, ((k - 1) * length(lambda) + 1):(k * length(lambda))] = y.temp
    }
  }
  # true leave-one-out risk estimate
  risk.alo = 1 / n * colSums((y.alo -
                                matrix(rep(y, dim(
                                  param
                                )[1]), ncol = dim(param)[1])) ^ 2)
  # return risk estimate
  return(risk.alo)
}

# compare the result
risk.alo = ElasticNet_ALO(X, y, param, alpha, lambda)
risk.alo.chol = ElasticNet_ALO_Chol(X, y, param, alpha, lambda)
risk.alo.block = ElasticNet_ALO_Block(X, y, param, alpha, lambda)

# plot
plot(risk.alo, risk.alo.block)

# compare the time
library(microbenchmark)
microbenchmark(
  ElasticNet_ALO(X, y, param, alpha, lambda),
  ElasticNet_ALO_Chol(X, y, param, alpha, lambda),
  ElasticNet_ALO_Block(X, y, param, alpha, lambda),
  times = 5
)