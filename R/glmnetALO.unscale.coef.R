glmnetALO.unscale.coef = function(mean_X, sd_X, a0, beta, family) {
  if (family == 3) {
    return(NULL)
  } else {
    beta = beta * sd_X
    a0 = as.vector(a0 + mean_X %*% beta)
  }
  
  return(list(a0 = a0, beta = beta))
}