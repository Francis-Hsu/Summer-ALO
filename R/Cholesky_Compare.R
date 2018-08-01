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
p =1000
k = 300
log10.lambda = seq(log10(1E-3), log10(5E-2), length.out = 50)
lambda = 10 ^ log10.lambda
lambda = sort(lambda, decreasing = TRUE)
alpha = 1
param = data.frame(alpha = numeric(0),
                   lambda = numeric(0))
for (i in 1:length(alpha)) {
  for (j in 1:length(lambda)) {
    param[j + (i - 1) * length(lambda), c('alpha', 'lambda')] = c(alpha[i], lambda[j])
  }
}
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

# normal ALO
# compute the scale parameter for y
sd.y = as.numeric(sqrt(var(y) * length(y) / (length(y) - 1)))
y.scaled = y / sd.y
X.scaled = X / sd.y
# find the ALO prediction
y.alo = matrix(ncol = dim(param)[1], nrow = n)
starttime = proc.time() # count time
for (k in 1:length(alpha)) {
  # build the full data model
  model = glmnet(
    x = X.scaled,
    y = y.scaled,
    family = "gaussian",
    alpha = alpha[k],
    lambda = lambda / sd.y ^ 2,
    thresh = 1E-14,
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
  # print middle result
  print(
    paste(
      k,
      " alphas have beed calculated. ",
      "On average, every alpha needs ",
      round((proc.time() - starttime)[3] / k, 2),
      " seconds."
    )
  )
}

# Cholesky decomposition ALO
# compute the scale parameter for y
sd.y = as.numeric(sqrt(var(y) * length(y) / (length(y) - 1)))
y.scaled = y / sd.y
X.scaled = X / sd.y
# find the ALO prediction
y.alo = matrix(ncol = dim(param)[1], nrow = n)
starttime = proc.time() # count time
XtX = t(cbind(1, X)) %*% cbind(1, X)
for (k in 1:length(alpha)) {
  # build the full data model
  model = glmnet(
    x = X.scaled,
    y = y.scaled,
    family = "gaussian",
    alpha = alpha[k],
    lambda = lambda / sd.y ^ 2,
    thresh = 1E-14,
    intercept = TRUE,
    standardize = FALSE,
    maxit = 1000000
  )
  # find the prediction for each alpha value
  if (alpha[k] == 1) {
    L = matrix(ncol = 0, nrow = 0)
    idx_old = numeric(0)
    for (j in 1:length(lambda)) {
      print(j)
      update = ElasticNetALO_CholUpdate(as.vector(model$beta[, j]),
                                        model$a0[j] * sd.y,
                                        X,
                                        y,
                                        lambda[j],
                                        alpha[k],
                                        L,
                                        idx_old,
                                        XtX)
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
  
  
  # print middle result
  print(
    paste(
      k,
      " alphas have beed calculated. ",
      "On average, every alpha needs ",
      round((proc.time() - starttime)[3] / k, 2),
      " seconds."
    )
  )
}

