library(alocv)
library(rbenchmark)

N = c(300, 500, 1000, 2500, 5000, 10000)
P = c(100, 800, 1200, 2000, 2500, 10000)
K = c(60, 500, 800, 1200, 2000, 2500)
tune_param = 10^seq(-2, -1, length.out = 25)
a = 0.5

m = length(N)
bmk = data.frame()
for(i in 1:m) {
  n = N[i]
  p = P[i]
  k = K[i]
  
  true_beta = rnorm(p, 0, 1)
  true_beta[-(1:k)] = 0
  X = matrix(rnorm(n * p, 0, sqrt(1 / k)), n, p)
  y = X %*% true_beta + rnorm(n, 0, 0.5)
  y[y >= 0] = sqrt(y[y >= 0])
  y[y < 0] = -sqrt(-y[y < 0])
  
  glm_obj = glmnet(X, y, lambda = tune_param, alpha = a, intercept = T, standardize = T)
  b = benchmark(cv.glmnet(X, y, nfolds = 5, alpha = a, lambda = tune_param, intercept = T, standardize = T), 
                glmnetALO(X, y, glm_obj, a, standardize = T), 
                columns = c("test", "elapsed", "relative"),
                replications = 10)
  bmk = rbind(bmk, b)
}
bmk = cbind(bmk[seq(2, 2 * m, 2 ),], bmk[seq(1, 2 * m, 2), ])

onefit = data.frame()
for(i in 1:m) {
  n = N[i]
  p = P[i]
  k = K[i]
  
  true_beta = rnorm(p, 0, 1)
  true_beta[-(1:k)] = 0
  X = matrix(rnorm(n * p, 0, sqrt(1 / k)), n, p)
  y = X %*% true_beta + rnorm(n, 0, 0.5)
  y[y >= 0] = sqrt(y[y >= 0])
  y[y < 0] = -sqrt(-y[y < 0])
  
  temp = benchmark(glmnet(X, y, lambda = tune_param, alpha = a, intercept = T, standardize = T),
                   columns = c("elapsed"),
                   replications = 10)
  onefit = rbind(onefit, temp)
}
colnames(onefit) = "one.fit"
bmk = cbind(N, P, bmk, N * onefit / 10)
rownames(bmk) = NULL
bmk