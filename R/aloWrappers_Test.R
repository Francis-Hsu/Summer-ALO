library(glmnet)
library(MASS)
library(Rcpp)
sourceCpp("glmnetALO.cpp")
source("aloWrappers.R")

######## Logistic ########
n = 300
p = 100
k = 60
true_beta = rnorm(p, 0, 4)
true_beta[-(1:k)] = 0

X = matrix(rnorm(n * p, 0, 1), n, p)
y = rbinom(n, size = 1, prob = 1 / (1 + exp(-X %*% true_beta)))

# par(mfrow = c(3, 3))

for(a in c(0.5)) {
  CV_bin = cv.glmnet(X, y, family = "binomial", nfolds = n, grouped = F, 
                     intercept = F, standardize = F, alpha = a, type.measure = "mae")
  ALO_bin = glmnetALO.Vanilla(X, y, a, CV_bin$glmnet.fit, type.measure = "mae")
  
  plot(CV_bin)
  par(new = T)
  plot(ALO_bin)
}

######## Poisson ########
n = 300
p = 100
k = 50
true_beta = rnorm(p, 0, 0.25)
true_beta[-(1:k)] = 0

X = matrix(rnorm(n * p, 0, 0.5), n, p)
y = rpois(n, lambda = exp(X %*% true_beta))
y = y / c(sd(y) * sqrt(n - 1) / sqrt(n))

tune_param = 10^seq(-2, -0.5, length.out = 25)

par(mfrow = c(3, 3))
for(a in seq(0, 1, length.out = 9)) {
  CV_pois = cv.glmnet(X, y, family = "poisson", lambda = tune_param, nfolds = n, grouped = F, 
                      intercept = F, standardize = F, alpha = a, type.measure = "mse")
  
  y_alo = glmnetALO.Vanilla(X, y, a, CV_pois$glmnet.fit)
}

# still working...
######## Multinomial ########
n = 300
p = 100
k = 60
num_class = 5

beta = matrix(rnorm(num_class * p, mean = 0, sd = 1), ncol = num_class)
beta[(k + 1):p,] = 0
intercept = rnorm(num_class, mean = 0, sd = 1)
X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
y.linear = matrix(rep(intercept, n), ncol = num_class, byrow = TRUE) +
  X %*% beta
prob = exp(y.linear) / matrix(rep(rowSums(exp(y.linear)), num_class), ncol =
                                num_class)
y.mat = t(apply(prob, 1, function(x)
  rmultinom(1, 1, prob = x))) # N * K matrix (N - #obs, K - #class)
y.num = apply(y.mat == 1, 1, which) # vector
y = factor(y.num, levels = seq(1:num_class))

a = 0.5

CV_ml = cv.glmnet(X, y, family = "multinomial", nlambda = 25, nfolds = n, grouped = F, 
                  intercept = T, standardize = F, alpha = a, type.measure = "mse")
y_alo = glmnetALO.Vanilla(X, y, a, CV_ml$glmnet.fit)


plot(rev(log10(CV_pois$glmnet.fit$lambda)), rev(CV_pois$cvm), 
     xlab = expression(lambda), ylab = "Risk", main = bquote(paste(alpha , " = ", .(a))), type = "l", lwd = 2, col = "darkorange")
lines(rev(log10(CV_pois$glmnet.fit$lambda)), rev(err), type = "b", pch = 4, lwd = 2, col = 4)