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
  mat I_n = eye<mat>(X.n_rows, X.n_rows);
  mat J = I_n - XE * inv_sympd(XE.t() * XE) * XE.t();
  vec y_alo = y - theta /  diagvec(J);
  
  return y_alo;
}

//[[Rcpp::export]]
vec elnetALO(const vec &beta, const mat &X, const vec &y, const double &lambda, const double &alpha) {
  // double sy = stddev(y, 1);
  vec yhat = X * beta;
  uvec E = find(abs(beta) >= 1e-8);
  mat XE = X.cols(E);
  mat hessR = (1 - alpha) * lambda * eye<mat>(E.n_elem, E.n_elem);
  mat H = XE * inv_sympd(XE.t() * XE / X.n_rows + hessR) * XE.t();
  vec y_alo = yhat + diagvec(H) % (yhat - y) / (X.n_rows - diagvec(H));
  
  return y_alo;
}

//[[Rcpp::export]]
vec lognetALO(const vec &beta, const mat &X, const vec &y, const double &lambda, const double &alpha) {
  vec yhat = X * beta;
  vec eXb = exp(yhat);
  uvec E = find(abs(beta) >= 1e-8);
  mat XE = X.cols(E);
  mat hessR = (1 - alpha) * lambda * eye<mat>(E.n_elem, E.n_elem);
  mat H = XE * inv_sympd(XE.t() * diagmat(eXb / square(1.0 + eXb)) * XE / X.n_rows + hessR) * XE.t();
  vec y_alo = yhat + diagvec(H) % (eXb / (1.0 + eXb) - y) / (X.n_rows - diagvec(H) % (eXb / square(1.0 + eXb)));
  
  return y_alo;
}

//[[Rcpp::export]]
// not working
mat multnetALO(const mat &B, const mat &X, const mat &Y, const double &lambda, const double &alpha) {
  mat XB = X * B;
  mat eXB = exp(XB);
  vec eXBRowSum = sum(eXB, 1);
  mat XB_alo = XB;

  vec eXb;
  vec lossSecDev;
  uvec E;
  mat XE;
  mat hessR;
  mat H;
  for(uword i = 0; i < B.n_cols; i++) {
    eXb = eXB.col(i);
    lossSecDev = eXb % (eXBRowSum - eXb) / square(eXBRowSum);
    E = find(abs(B.col(i)) >= 1e-8);
    XE = X.cols(E);
    hessR = (1 - alpha) * lambda * eye<mat>(E.n_elem, E.n_elem);
    H = XE * inv_sympd(XE.t() * diagmat(lossSecDev) * XE / X.n_rows + hessR) * XE.t();
    XB_alo.col(i) += diagvec(H) % (eXb / eXBRowSum - Y.col(i)) / (X.n_rows - diagvec(H) % lossSecDev);
  }
  
  return XB_alo;
}

//[[Rcpp::export]]
vec fishnetALO(const vec &beta, const mat &X, const vec &y, const double &lambda, const double &alpha) {
  vec yhat = X * beta;
  vec eXb = exp(yhat);
  uvec E = find(abs(beta) >= 1e-8);
  mat XE = X.cols(E);
  mat hessR = (1 - alpha) * lambda * eye<mat>(E.n_elem, E.n_elem);
  mat H = XE * inv_sympd(XE.t() * diagmat(eXb) * XE / X.n_rows + hessR) * XE.t();
  vec y_alo = yhat + diagvec(H) % (eXb - y) / (X.n_rows - diagvec(H) % eXb);
  
  return y_alo;
}

//[[Rcpp::export]]
vec fusedALO(const vec &beta, const vec &u, const mat &X, const vec &y, const mat &D, const double &lambda) {
  vec yhat = X * beta;
  //uvec mE = find(abs(diff(beta)) <= 1e-4);
  uvec mE = find(abs(abs(u) - lambda) >= 1e-4);
  mat DmE = D.rows(mE);
  mat B = null(DmE);
  mat A = X * B;
  mat H = A * pinv(A);
  vec y_alo = yhat + diagvec(H) % (yhat - y) / (1 - diagvec(H));
  
  return y_alo;
}

//[[Rcpp::export]]
vec slopeALO(const vec &beta, const mat &X, const vec &y, const vec &lambda) {
  vec yhat = X * beta;
  vec theta = y - yhat;
  vec Xtheta = X.t() * theta;
  
  uvec E = sort(find(beta != 0), "descend");
  uvec k = sort_index(abs(Xtheta), "descend");
  vec s = sign(X.elem(k));
  mat S(E.n_elem, E.n_elem, fill::zeros);
  for(uword u = 0; u < E.n_elem; u++) {
    for(uword v = 0; v < E.n_elem; v++) {
      if(E(v) <= E(u)) {
        S(u, v) = s(k(E(v)));
      }
    }
  }
  mat A = X.cols(E) * S;
  mat H = A * inv_sympd(A.t() * A) * A.t();
  vec y_alo = y + (y - yhat) / (1 - diagvec(H));
  
  return y_alo;
}