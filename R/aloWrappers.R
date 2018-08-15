lassoALO.Woodbury = function(X, y, glm_obj) {
  # extract stuffs from the glmnet object
  glm_lambda = glm_obj$lambda # assume at least 2 lambdas
  glm_beta = glm_obj$beta
  glm_a0 = glm_obj$a0
  M = length(glm_lambda)
  
  # check intercept
  if(any(glm_a0 != 0)) {
    X = cbind(1, X)
    glm_beta = rbind(glm_a0, glm_beta)
  }
  
  # find update/downdate indices
  glm_active_idx = apply(glm_beta, 2, function(b) { unname(which(abs(b) >= 1e-8)) - 1 })
  glm_add_idx = lapply(2:M, function(i) {
    setdiff(glm_active_idx[[i]], glm_active_idx[[i - 1]])
  })
  glm_rmv_idx = lapply(2:M, function(i) {
    setdiff(glm_active_idx[[i - 1]], glm_active_idx[[i]])
  })
  alo_update = lassoALOMIL(X, y, glm_beta, glm_active_idx, glm_add_idx, glm_rmv_idx)
  mse = colMeans(alo_update^2)
  
  return(mse)
}

lassoALO.Vanilla = function(X, y, glm_obj) {
  glm_lambda = glm_obj$lambda # assume at least 2 lambdas
  glm_beta = glm_obj$beta
  glm_a0 = glm_obj$a0
  
  # check intercept
  if(any(glm_a0 != 0)) {
    X = cbind(1, X)
    glm_beta = rbind(glm_a0, glm_beta)
  }
  
  Y_alo = foreach(i = 1:length(glm_lambda), .combine = cbind) %do% {
    # temp_lambda = glm_lambda[i]
    temp_beta = glm_beta[, i]
    alo_pred = lassoALO(temp_beta, X, y) - y
  }
  
  return(colMeans(Y_alo^2))
}

elnetALO.Approx = function(X, y, alpha, glm_obj) {
  # extract stuffs from the glmnet object
  glm_lambda = glm_obj$lambda # assume at least 2 lambdas
  glm_beta = glm_obj$beta
  glm_a0 = glm_obj$a0
  M = length(glm_lambda)
  
  # check intercept
  if(any(glm_a0 != 0)) {
    X = cbind(1, X)
    glm_beta = rbind(glm_a0, glm_beta)
  }
  
  # find update/downdate indices
  glm_active_idx = apply(glm_beta, 2, function(b) { unname(which(abs(b) >= 1e-8)) - 1 })
  glm_add_idx = lapply(2:M, function(i) {
    setdiff(glm_active_idx[[i]], glm_active_idx[[i - 1]])
  })
  glm_rmv_idx = lapply(2:M, function(i) {
    setdiff(glm_active_idx[[i - 1]], glm_active_idx[[i]])
  })
  alo_update = elnetALOApprox(X, y, glm_beta, glm_lambda, alpha, glm_active_idx, glm_add_idx, glm_rmv_idx)
  mse = colMeans(sweep(alo_update, 1, y)^2)
  
  return(mse)
}

elnetALO.Vanilla = function(X, y, alpha, glm_obj) {
  glm_lambda = glm_obj$lambda # assume at least 2 lambdas
  glm_beta = glm_obj$beta
  glm_a0 = glm_obj$a0
  
  # check intercept
  if(any(glm_a0 != 0)) {
    X = cbind(1, X)
    glm_beta = rbind(glm_a0, glm_beta)
  }
  
  Y_alo = foreach(i = 1:length(glm_lambda), .combine = cbind) %do% {
    # temp_lambda = glm_lambda[i]
    temp_beta = glm_beta[, i]
    temp_lambda = glm_lambda[i]
    alo_pred = elnetALO(temp_beta, X, y, temp_lambda, alpha) - y
  }
  
  return(colMeans(Y_alo^2))
}