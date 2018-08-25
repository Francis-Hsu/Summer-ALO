glmnetALO = function(X, y, alpha, glm_obj, type.measure = "mse", standardize = F) {
  # coerce format
  X = as.matrix(X)
  y = as.matrix(y)
  
  # extract useful stuffs from glmnet object
  glm_family = class(glm_obj)[1]
  glm_lambda = glm_obj$lambda # assume at least 2 lambdas
  glm_beta = glm_obj$beta # estimates
  glm_a0 = glm_obj$a0 # intercepts
  nz = glm_obj$df
  K = ncol(y) # number of classes, for multinomial regression
  
  # response type of the glmnet object
  # support "gaussian", "binomial", "poisson" and "multinomial" for now
  family_idx = switch(glm_family,
                      elnet = 0,
                      lognet = 1,
                      fishnet = 2,
                      multnet = 3,
                      stop("Unsupported regression family, must be \"gaussian\", \"binomial\", \"poisson\", or \"multinomial\""))
  intercept_flag = any(glm_a0 != 0) # check intercept
  
  # Preprocessing
  # standardize input and restandardize coefficients
  # what about categorical variables?
  if (standardize) {
    mean_X = colMeans(X)
    sd_X = sqrt(colSums(X^2 / n) - colSums(X / n)^2)
    X = scale(X, scale = sd_X)
    glm_rescale = glmnetALO.unscale.coef(mean_X, sd_X, glm_a0, glm_beta, family_idx)
    glm_a0 = glm_rescale$a0
    glm_beta = glm_rescale$beta
  }
  
  # prepare beta and intercept
  if (family_idx == 3) {
    X = multinetExpand(X, K)
    glm_beta = do.call(rbind, glm_beta)
    if (intercept_flag) {
      X = cbind(do.call(rbind, replicate(nrow(y), diag(K), simplify = FALSE)), X)
      glm_beta = rbind(glm_a0, glm_beta)
    }
  } else {
    if (intercept_flag) {
      X = cbind(1, X)
      glm_beta = rbind(glm_a0, glm_beta)
    }
  }
  # glm_beta = unname(glm_beta) # Drop the names
  
  # identify active set, find variables to drop/remove
  M = length(glm_lambda)
  glm_active_set = lapply(1:M, function(i) { which(abs(glm_beta[, i]) >= 1e-8) - 1 })
  glm_add_idx = lapply(2:M, function(i) {
    setdiff(glm_active_set[[i]], glm_active_set[[i - 1]])
  })
  glm_add_idx = append(glm_add_idx, list(glm_active_set[[1]]), 0)
  glm_rmv_idx = lapply(2:M, function(i) {
    setdiff(glm_active_set[[i - 1]], glm_active_set[[i]])
  })
  
  # compute ALO and corresponding risk
  y_alo = glmnetALODirect(X, as.matrix(y), glm_beta, glm_lambda, alpha, 
                          glm_add_idx, glm_rmv_idx, 
                          family_idx, intercept_flag)
  alo_risk = glmnetALO.risk(y_alo, y, family_idx, type.measure)
  alom = alo_risk$alom
  alosd = alo_risk$alosd
  
  # find optimum values of lambda
  idmin = which.min(alo_risk$alom)
  lambda.min = glm_lambda[idmin]
  semin = alom[idmin] + alosd[idmin]
  id1se = which(alom <= semin)
  lambda.1se = max(glm_lambda[id1se], na.rm = TRUE)
  
  # create and return glmnetALO object
  # S4 later?
  rtn_obj = list(yALO = y_alo, 
                 riskm = alom, risksd = alosd, riskup = alom + alosd, risklo = alom - alosd,
                 nzeros = nz, alpha = alpha,
                 lambda = glm_lambda, lambda.min = lambda.min, lambda.1se = lambda.1se, 
                 family = glm_family, type.measure = type.measure)
  attr(rtn_obj, "class") = "glmnetALO"
  return(rtn_obj)
}