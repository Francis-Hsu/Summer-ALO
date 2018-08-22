#include <RcppArmadillo.h>
#include <math.h>

using namespace Rcpp;
using namespace arma;

//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

//[[Rcpp::export]]
vec lassoALO(const vec &beta, const mat &X, const vec &y) {
  vec theta = y - X * beta;
  uvec E = find(abs(beta) >= 1e-8);
  mat XE = X.cols(E);
  mat R = chol(XE.t() * XE);
  mat XER = solve(trimatl(R.t()), XE.t());
  vec H = sum(square(XER.t()), 1);
  vec yAlo = y - theta / (1 - H);
  
  return yAlo;
}

//[[Rcpp::export]]
mat elnetALO(const mat &XE, const mat &yHat, const mat &y, const double &lambda, const double &alpha, const bool intercept) {
  uword n = XE.n_rows; // # of samples
  uword p = XE.n_cols; // # of active variables
  
  mat hessR = (1 - alpha) * lambda * eye<mat>(p, p);
  mat F = inv_sympd(XE.t() * XE / n + hessR);
  vec H = sum((XE * F) % XE, 1);
  mat yALO = yHat + H % (yHat - y) / (n - H);
  
  return yALO;
}

//[[Rcpp::export]]
mat lognetALO(const mat &XE, const mat &yHat, const mat &y, const double &lambda, const double &alpha, const bool intercept) {
  uword n = XE.n_rows; // # of samples
  uword p = XE.n_cols; // # of active variables
  
  vec eXb = exp(yHat);
  mat hessR = (1 - alpha) * lambda * eye<mat>(p, p);
  mat F = inv_sympd(XE.t() * diagmat(eXb / square(1.0 + eXb)) * XE / n + hessR);
  vec H = sum((XE * F) % XE, 1);
  mat yALO = yHat + H % (eXb / (1.0 + eXb) - y) / (n - H % (eXb / square(1.0 + eXb)));
  
  return yALO;
}

//[[Rcpp::export]]
mat fishnetALO(const mat &XE, const mat &yHat, const mat &y, const double &lambda, const double &alpha, const bool intercept) {
  uword n = XE.n_rows; // # of samples
  uword p = XE.n_cols; // # of active variables
  
  vec eXb = exp(yHat);
  mat hessR = (1 - alpha) * lambda * eye<mat>(p, p);
  mat F = inv_sympd(XE.t() * diagmat(eXb) * XE / n + hessR);
  vec H = sum((XE * F) % XE, 1);
  mat yALO = yHat + H % (eXb - y) / (n - H % eXb);
  
  return yALO;
}

// [[Rcpp::export]]
mat multinetALO(const mat &XESp, const mat &yHat, const mat &y, const double &lambda, const double &alpha, const bool intercept) {
  // find out the dimension of X
  uword K = yHat.n_cols; // K
  uword n = yHat.n_rows;
  uword p = intercept ? XESp.n_cols - 1 : XESp.n_cols; // # of active variables, including intercepts (if any)
  
  // compute vector A(beta) and matrix D(beta)
  vec A(n * K, fill::none);
  mat D(n * K, n * K, fill::zeros);
  vec tempA;
  for (uword i = 0; i < n; ++i) {
    // use span() instead
    // check out normalise() later
    tempA = exp(yHat.rows(span(i * K, i * K + K - 1)));
    A(span(i * K, i * K + K - 1)) = tempA / sum(tempA);
    D(span(i * K, i * K + K - 1), span(i * K, i * K + K - 1)) = diagmat(tempA) - tempA * tempA.t();
  }
  
  // compute hessR
  mat hessR = n * lambda * (1 - alpha) * eye<mat>(p * K, p * K);
  if (intercept) {
    hessR(span(0, K - 1), span(0, K - 1)) = zeros<mat>(K, K); // no penalty for intercepts
  }
  // compute matrix K(beta) and its inverse
  mat K_inv = pinv(XESp.t() * D * XESp + hessR, 0);
  
  // do leave-i-out prediction
  mat aloUpdate(n, K, fill::none);
  for (uword i = 0; i < n; ++i) {
    // find the X_i and y_i
    uvec idx = regspace<uvec>(i * K, 1, (i + 1) * K - 1);
    mat X_i = XESp.rows(idx);
    vec y_i = conv_to<vec>::from(y.row(i));
    
    // find A_i
    vec A_i = A(idx);
    
    // compute XKX
    mat XKX = X_i * K_inv * X_i.t();
    
    // compute the inversion of diag(A)-A*A^T
    mat middle_inv = pinv(diagmat(A_i) - A_i * A_i.t(), 0);
    
    // compute the leave-i-out prediction
    aloUpdate.row(i) = XKX * (A_i - y_i) - XKX * pinv(-middle_inv + XKX, 0) * XKX * (A_i - y_i);
  }
  mat yALO = yHat + aloUpdate;
  
  return(yALO);
}

//[[Rcpp::export]]
// Approx matrix inverse through iterating method
mat glmnetALOVanilla(const mat &X, const vec &y, const sp_mat &beta, const vec &lambda, const double &alpha,
                     const field<uvec> &activeList, const field<uvec> &addList, const field<uvec> &dropList,
                     const int family, const bool intercept) {
  // assign function pointer
  mat (*netPtr)(const mat &, const mat &, const mat &, const double &, const double &, const bool) = NULL;
  if (family == 0) {
    netPtr = &elnetALO;
  } else if (family == 1) {
    netPtr = &lognetALO;
  } else if (family == 2) {
    netPtr = &fishnetALO;
  } else if (family == 3) {
    netPtr = &multinetALO;
  }
  
  // preamble
  uvec currXIdx = activeList(0);
  mat aloUpdate(X.n_rows, beta.n_cols, fill::zeros); // matrix storing ALO estimates
  vec yHat = X * beta.col(0);
  mat XE = X.cols(currXIdx);
  aloUpdate.col(0) = netPtr(XE, yHat, y, lambda(0), alpha, intercept);
  
  // update through all lambdas
  uvec toAdd;
  uvec toDrop;
  uvec XEcolIdx;
  uvec dropIdx;
  for (uword i = 1; i < beta.n_cols; i++) {
    // checkUserInterrupt();
    toAdd = addList(i - 1); // columns to add
    toDrop = dropList(i - 1); // columns to drop
    
    if (currXIdx.n_elem > 0) {
      // remove variables
      if (toDrop.n_elem > 0) {
        XEcolIdx = regspace<uvec>(0, currXIdx.n_elem - 1);
        dropIdx.zeros(toDrop.n_elem);
        for (uword j = 0; j < toDrop.n_elem; j++) {
          dropIdx(j) = conv_to<uword>::from(find(currXIdx == toDrop(j), 1, "first"));
        }
        dropIdx = sort(dropIdx, "descend");
        for (uword j = 0; j < toDrop.n_elem; j++) {
          currXIdx.shed_row(dropIdx(j));
          XEcolIdx.shed_row(dropIdx(j));
        }
        XE = XE.cols(XEcolIdx);
      }
      // add variables
      if (toAdd.n_elem > 0) {
        currXIdx = join_cols(currXIdx, toAdd);
        XE = join_rows(XE, X.cols(toAdd));
      }
    } else {
      currXIdx = toAdd;
      XE = X.cols(toAdd);
    }
    yHat = X * beta.col(i);
    aloUpdate.col(i) = netPtr(XE, yHat, y, lambda(i), alpha, intercept);
  }
  
  return aloUpdate;
}

// [[Rcpp::export]]
mat multinetExpand(const mat &X, const uword K) {
  uword n = X.n_rows;
  uword p = X.n_cols;
  mat XSp(n * K, p * K, fill::zeros);
  uvec rowID, colID;
  for (uword i = 0; i < K; i++) {
    rowID = regspace<uvec>(0, K, n * K - K) + i;
    colID = regspace<uvec>(p * i, p * i + p - 1);
    XSp.submat(rowID, colID) = X;
  }
  
  return XSp;
}