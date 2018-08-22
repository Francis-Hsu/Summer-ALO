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

glmnetALO.Vanilla = function(X, y, alpha, glm_obj, type.measure = "mse", standardize = F) {
  # preamble
  glm_family = class(glm_obj)[1]
  glm_lambda = glm_obj$lambda # assume at least 2 lambdas
  glm_beta = glm_obj$beta
  glm_a0 = glm_obj$a0
  nz = glm_obj$df
  K = ncol(y) # number of classes, for multinomial regression
  
  # response type of the glmnet object
  # support "gaussian", "binomial", and "poisson" for now
  family_idx = switch(glm_family,
                      elnet = 0,
                      lognet = 1,
                      fishnet = 2,
                      multnet = 3,
                      stop("Unsupported regression family, must be \"gaussian\", \"binomial\", \"poisson\", or \"multinomial\""))
  intercept_flag = any(glm_a0 != 0) # check intercept
  
  # Preprocessing
  if (family_idx == 3) {
    X = multinetExpand(X, K)
    glm_beta = do.call(rbind, glm_beta)
    if (intercept_flag) {
      X = cbind(do.call(rbind, replicate(nrow(y), diag(K), simplify = FALSE)), XSp)
      glm_beta = rbind(model$a0, glm_beta)
    }
  } else {
    if (intercept_flag) {
      X = cbind(1, X)
      glm_beta = rbind(glm_a0, glm_beta)
    }
  }
  glm_beta = unname(glm_beta) # Drop the names
  
  if (standardize) {
    glm_beta = glm_beta * sqrt(colSums(X^2 / n) - colSums(X / n)^2) # fix for multinet later
  }
  
  # lists for update/downdate
  M = length(glm_lambda)
  glm_active_idx = lapply(1:M, function(i) { which(abs(glm_beta[, i]) >= 1e-8) - 1 })
  glm_add_idx = lapply(2:M, function(i) {
    setdiff(glm_active_idx[[i]], glm_active_idx[[i - 1]])
  })
  glm_rmv_idx = lapply(2:M, function(i) {
    setdiff(glm_active_idx[[i - 1]], glm_active_idx[[i]])
  })
  
  # compute ALO and corresponding risk
  y_alo = glmnetALOVanilla(X, y, glm_beta, glm_lambda, alpha, 
                           glm_active_idx, glm_add_idx, glm_rmv_idx, 
                           family_idx, intercept_flag)
  alo_risk = aloError(y_alo, y, family_idx, type.measure)
  alom = alo_risk$alom
  alosd = alo_risk$alosd
  
  # find optimum values of lambda
  idmin = which.min(alo_risk$alom)
  lambda.min = glm_lambda[idmin]
  semin = alom[idmin] + alosd[idmin]
  id1se = which(alom <= semin)
  lambda.1se = max(glm_lambda[id1se], na.rm = TRUE)
  
  # create and return glmnetALO object
  rtn_obj = list(yALO = y_alo, 
                 riskm = alom, risksd = alosd, riskup = alom + alosd, risklo = alom - alosd,
                 nzeros = nz, alpha = alpha,
                 lambda = glm_lambda, lambda.min = lambda.min, lambda.1se = lambda.1se, 
                 family = glm_family, type.measure = type.measure)
  attr(rtn_obj, "class") = "glmnetALO"
  return(rtn_obj)
}

# add weights support later?
aloError = function(yalo, y, family, type.measure) {
  if (family == 0) {
    errmat = sweep(yalo, 1, y)
    aloraw = switch(type.measure,
                    mse = errmat^2,
                    deviance = errmat^2,
                    mae = abs(errmat))
  } else if (family == 1) {
    y = as.factor(y)
    nc = as.integer(length(table(y)))
    y = diag(nc)[as.numeric(y), ]
    predmat = 1 / (1 + exp(-yalo))
    prob_min = 1e-05
    prob_max = 1 - prob_min
    aloraw = switch(type.measure,
                    mse = (y[, 1] - (1 - predmat))^2 + (y[, 2] - predmat)^2,
                    mae = abs(y[, 1] - (1 - predmat)) + abs(y[, 2] - predmat),
                    class = y[, 1] * (predmat > 0.5) + y[, 2] * (predmat <=0.5),
                    deviance = {
                      predmat = pmin(pmax(predmat, prob_min), prob_max)
                      lp = y[, 1] * log(1 - predmat) + y[, 2] * log(predmat)
                      ly = log(y)
                      ly[y == 0] = 0
                      ly = drop((y * ly) %*% c(1, 1))
                      2 * (ly - lp)
                    })
  } else if (family == 2) {
    errmat = sweep(exp(yalo), 1, y)
    aloraw = switch(type.measure,
                    mse = errmat^2,
                    mae = abs(errmat))
  }
  alom = colMeans(aloraw)
  alosd = sqrt(colMeans(scale(aloraw, alom, FALSE)^2, na.rm = TRUE) / (length(y) - 1))
  
  return(list(alom = alom, alosd = alosd))
}

# class plot method later
plot.glmnetALO = function(obj, ...) {
  # extract useful stuffs
  obj_measure = obj$type.measure
  obj_family = obj$family
  obj_riskm = obj$riskm
  obj_riskup = obj$riskup
  obj_risklo = obj$risklo
  obj_lambda = obj$lambda
  obj_lambda.min = obj$lambda.min
  obj_lambda.1se = obj$lambda.1se
  obj_nz = obj$nzeros
  
  log_param = log(obj_lambda)
  
  xlab = "log(Lambda)"
  ylab = switch(obj_measure,
                mse = "MSE",
                mae = "MAE",
                deviance = "Deviance",
                class = "Class")
  plot.args = list(x = log(obj_lambda),
                   y = obj_riskm,
                   ylim = range(obj_riskup, obj_risklo), 
                   xlab = xlab, 
                   ylab = ylab, 
                   type = "n")
  new.args = list(...)
  if (length(new.args)){ 
    plot.args[names(new.args)] = new.args
  }
  do.call("plot", plot.args)
  
  # plot error bars
  barw = diff(range(log_param)) * 0.01
  segments(log_param, obj_riskup, log_param, obj_risklo, col="darkgrey")
  segments(log_param - barw, obj_riskup, log_param + barw, obj_riskup, col="darkgrey")
  segments(log_param - barw, obj_risklo, log_param + barw, obj_risklo, col="darkgrey")
  
  # plot risks
  points(log_param, obj_riskm, pch = 20, col = "blue")
  axis(side = 3, at = log_param, labels = paste(obj_nz), tick = FALSE, line = 0)
  abline(v = log(obj_lambda.min), lty = 3)
  abline(v = log(obj_lambda.1se), lty = 3)
  invisible()
}