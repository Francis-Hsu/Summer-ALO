#include <RcppArmadillo.h>
#include <math.h>

using namespace Rcpp;
using namespace arma;

// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
vec ElasticNetALO(vec beta, double intercept, 
                  mat X, vec y, 
                  double lambda, double alpha) {
  // find out the dimension of X
  double n = X.n_rows;
  double p = X.n_cols;
  // define full data set
  vec ones(n,fill::ones);
  mat X_full = X;
  X_full.insert_cols(0, ones);
  // define full beta (intercept & slope)
  vec beta_full(p + 1);
  beta_full(0) = intercept;
  beta_full(span(1, p)) = beta;
  // compute prediction
  vec y_hat = X_full * beta_full;
  // find active set
  uvec A = find(beta_full != 0);
  // compute matrix H
  mat H(n, n, fill::zeros);
  if(!A.is_empty()) {
    mat X_active = X_full.cols(A);
    mat R_diff2(A.n_elem, A.n_elem, fill::eye);
    R_diff2 = R_diff2 * n * lambda * (1-alpha);
    if(intercept != 0) {
      R_diff2(0,0) = 0;
    }
    mat L = chol(X_active.t() * X_active + R_diff2).t();
    mat L_inv = inv(L);
    mat AE = L_inv * X_active.t();
    H = AE.t() * AE;
    // H = X_active * inv_sympd(X_active.t() * X_active + R_diff2) * X_active.t();
  }
  // compute the ALO prediction
  vec y_alo = y_hat + H.diag() % (y_hat - y) / (1-H.diag());
  return y_alo;
}

// [[Rcpp::export]]
vec LogisticALO(vec beta, double intercept, 
                mat X, vec y, 
                double lambda, double alpha) {
  // find out the dimension of X
  double n = X.n_rows;
  double p = X.n_cols;
  // define full data set
  vec ones(n,fill::ones);
  mat X_full = X;
  X_full.insert_cols(0, ones);
  // define full beta (intercept & slope)
  vec beta_full(p + 1);
  beta_full(0) = intercept;
  beta_full(span(1, p)) = beta;
  // compute linear prediction
  vec y_linear = X_full * beta_full;
  vec y_exp = exp(y_linear);
  // find active set
  uvec A = find(beta_full != 0);
  // compute matrix D
  mat D(n, n, fill::zeros);
  D.diag() = y_exp / ((1 + y_exp) % (1 + y_exp));
  // compute matrix H
  mat H(n, n, fill::zeros);
  if(!A.is_empty()) {
    mat X_active = X_full.cols(A);
    mat R_diff2(A.n_elem, A.n_elem, fill::eye);
    R_diff2 = R_diff2 * n * lambda * (1 - alpha);
    if(intercept != 0) {
      R_diff2(0,0) = 0;
    }
    mat L = chol(X_active.t() * D * X_active + R_diff2).t();
    mat L_inv = inv(L);
    mat AE = L_inv * X_active.t();
    H = AE.t() * AE;
  }
  // compute the ALO prediction
  vec y_alo_linear = y_linear + H.diag() % (y_exp / (1 + y_exp) - y) / (1-H.diag() % D.diag());
  vec y_alo = exp(y_alo_linear) / (1 + exp(y_alo_linear));
  return y_alo;
}

// [[Rcpp::export]]
vec PoissonALO(vec beta, double intercept, 
               mat X, vec y, 
               double lambda, double alpha) {
  // find out the dimension of X
  double n = X.n_rows;
  double p = X.n_cols;
  // define full data set
  vec ones(n,fill::ones);
  mat X_full = X;
  X_full.insert_cols(0, ones);
  // define full beta (intercept & slope)
  vec beta_full(p + 1);
  beta_full(0) = intercept;
  beta_full(span(1, p)) = beta;
  // compute linear prediction
  vec y_linear = X_full * beta_full;
  // find active set
  uvec A = find(beta_full != 0);
  // compute matrix D
  mat D(n, n, fill::zeros);
  D.diag() = exp(y_linear);
  // compute matrix H
  mat H(n, n, fill::zeros);
  if(!A.is_empty()) {
    mat X_active = X_full.cols(A);
    mat R_diff2(A.n_elem, A.n_elem, fill::eye);
    R_diff2 = R_diff2 * n * lambda * (1 - alpha);
    if(intercept != 0) {
      R_diff2(0,0) = 0;
    }
    mat L = chol(X_active.t() * D * X_active + R_diff2).t();
    mat L_inv = inv(L);
    mat AE = L_inv * X_active.t();
    H = AE.t() * AE;
  }
  // compute the ALO prediction
  vec y_alo_linear = y_linear + H.diag() % (exp(y_linear) - y) / (1-H.diag() % D.diag());
  vec y_alo = exp(y_alo_linear);
  return y_alo;
}

// [[Rcpp::export]]
mat MultinomialALO(mat beta, vec intercept, 
                   mat X, mat y, 
                   double lambda, double alpha) {
  // find out the dimension of X
  double n = X.n_rows; // N
  double p = X.n_cols; // P
  double num_class = y.n_cols; // K
  // define the output matrix
  mat y_alo_linear(n, num_class);
  mat y_alo_exp(n, num_class);
  mat y_alo(n, num_class);
  // define full data set
  vec ones(n,fill::ones);
  mat X_full = X; // N * (P + 1)
  X_full.insert_cols(0, ones);
  // define full beta (intercept & slope)
  mat beta_full(p + 1, num_class); // (P + 1) * K
  beta_full(0, span(0, num_class - 1)) = intercept.t();
  beta_full(span(1, p), span(0, num_class - 1)) = beta;
  // compute linear prediction and its exponential value
  mat y_linear = X_full * beta_full; // N * K
  mat y_exp = exp(y_linear); // N * K
  // compute the summation of all classes y_exp
  vec sum_y_exp = sum(y_exp, 1); // N * 1
  // for loop to compute the prediction under each class
  for (int k = 0; k < num_class; ++k) {
    // find active set
    uvec A = find(beta_full.col(k) != 0);
    // compute matrix D
    mat D(n, n, fill::zeros);
    D.diag() = y_exp.col(k) % (sum_y_exp - y_exp.col(k)) / (sum_y_exp % sum_y_exp);
    // compute matrix H
    mat H(n, n, fill::zeros);
    if(!A.is_empty()) {
      mat X_active = X_full.cols(A);
      mat R_diff2(A.n_elem, A.n_elem, fill::eye);
      R_diff2 = R_diff2 * n * lambda * (1 - alpha);
      if(intercept(k) != 0) {
        R_diff2(0,0) = 0;
      }
      mat L = chol(X_active.t() * D * X_active + R_diff2).t();
      mat L_inv = inv(L);
      mat AE = L_inv * X_active.t();
      H = AE.t() * AE;
    }
    // compute the ALO prediction
    y_alo_linear.col(k) = y_linear.col(k) + 
      H.diag() % (y_exp.col(k) / sum_y_exp - y.col(k)) / 
      (1 - H.diag() % D.diag());
    y_alo_exp.col(k) = exp(y_alo_linear.col(k));
  }
  // compute the summation of all classes y_exp
  vec sum_y_alo_exp = sum(y_alo_exp, 1); // N * 1
  // compute the alo prediction
  for (int k = 0; k < num_class; ++k) {
    y_alo.col(k) = y_alo_exp.col(k) / sum_y_alo_exp;
  }
  return y_alo;
}

// [[Rcpp::export]]
vec GenLASSOALO(vec beta, vec u, mat X, vec y, mat D, 
                double lambda, double tol) {
  // compute prediction
  vec y_hat = X * beta;
  // find complement of the active set
  uvec mE = find(abs(abs(u) - lambda) >= tol);
  // find the null space of D_{-E,*}
  mat VE = null(D.rows(mE));
  // find matrix H
  mat H = X * VE * pinv(X * VE);
  // compute the ALO prediction
  vec y_alo = y + (y_hat - y) / (1-H.diag());
  return y_alo;
}

// [[Rcpp::export]]
field<mat> CholeskyAdd(mat X, mat L, 
                       uvec idx_old, uvec idx_new) {
  // X - full symmetric matrix
  // L - upper triangle matrix of Cholesky decomposition for X[idx_old, idx_old]
  // idx_old - old index of L for the X[idx_old, idx_old]
  // idx_new - find the Cholesky decomposition for X[idx_new, idx_new], 
  //           where idx_old \in idx_new
  
  // if idx_old is empty, just find the Cholesky of the idx_new
  if (idx_old.is_empty() && !idx_new.is_empty()) {
    mat L_out = chol(X(idx_new,idx_new));
    field<mat> out(2);
    out(0) = L_out;
    out(1) = conv_to<mat>::from(idx_new);
    return out;
  }
  // find the rows that is needed to be added
  uvec idx_add;
  for (size_t i = 0; i < idx_new.n_elem; ++i) {
    uvec temp = find(idx_old == idx_new[i]);
    if (temp.is_empty()) {
      idx_add.resize(idx_add.n_elem + 1);
      idx_add(idx_add.n_elem - 1) = idx_new[i];
    }
  }
  // if idx_add is empty - no adding rows, then return matrix
  if (idx_add.is_empty()) {
    field<mat> out(2);
    out(0) = L; // upper triangle matrix
    out(1) = conv_to<mat>::from(idx_old);
    return out;
  }
  // update Cholesky decomposition
  mat A11 = X(idx_old, idx_old);
  mat A12 = X(idx_old, idx_add);
  mat A22 = X(idx_add, idx_add);
  mat S11 = L;
  mat S12 = solve(L.t(), A12);
  mat S22 = chol(A22 - S12.t() * S12);
  mat L_out(idx_new.n_elem, idx_new.n_elem, fill::zeros);
  L_out.submat(span(0, S11.n_rows - 1), span(0, S11.n_cols - 1)) = S11;
  L_out.submat(span(0, S12.n_rows - 1), span(S11.n_cols, L_out.n_cols - 1)) = S12;
  L_out.submat(span(S11.n_rows, L_out.n_rows - 1), span(S11.n_cols, L_out.n_cols - 1)) = S22;
  // find the output index for Cholesky decomposition
  uvec idx_out = idx_old;
  idx_out.resize(idx_new.n_elem);
  idx_out(span(idx_old.n_elem, idx_out.n_elem - 1)) = idx_add;
  // output
  field<mat> out(2);
  out(0) = L_out;
  out(1) = conv_to<mat>::from(idx_out);
  return out;
}

// [[Rcpp::export]]
field<mat> CholeskyDrop(mat X, mat L, 
                        uvec idx_old, uvec idx_new) {
  // X - full symmetric matrix
  // L - upper triangle matrix of Cholesky decomposition for X[idx_old, idx_old]
  // idx_old - old index of L for the X[idx_old, idx_old]
  // idx_new - find the Cholesky decomposition for X[idx_new, idx_new], 
  //           where idx_new \in idx_old
  
  // find one row that is needed to be droped
  uvec idx_drop;
  uword pos = 0;
  for (uword i = 0; i < idx_old.n_elem; ++i) {
    uvec temp = find(idx_new == idx_old(i));
    if (temp.is_empty()) {
      idx_drop = idx_old(i); // drop this row
      pos = i;
      break;
    }
  }
  // if idx_drop is empty - no dropping row, then return matrix
  if (idx_drop.is_empty()) {
    field<mat> out(2);
    out(0) = L; // upper triangle matrix
    out(1) = conv_to<mat>::from(idx_old);
    return out;
  } else if (pos == 0) { // drop the first row
    if (L.n_cols == 1) { // only have one row before dropping
      mat L_out(0,0);
      uvec idx_out;
      field<mat> out(2);
      out(0) = L_out;
      out(1) = conv_to<mat>::from(idx_out);
      return out;
    } else {
      mat S23 = L(pos, span(pos + 1, L.n_cols - 1));
      mat S33 = L(span(pos + 1, L.n_rows - 1), span(pos + 1, L.n_cols - 1));
      field<mat> out(2);
      out(0) = chol(S33.t() * S33 + S23.t() * S23);
      out(1) = conv_to<mat>::from(idx_old(span(pos + 1, idx_old.n_elem - 1)));
      return out;
    }
  } else if (pos == idx_old.n_elem - 1) { // drop the last row
    field<mat> out(2);
    out(0) = L(span(0, pos - 1), span(0, pos - 1)); // upper triangle matrix
    out(1) = conv_to<mat>::from(idx_old(span(0, pos - 1)));
    return out;
  }
  // update Cholesky decomposition
  mat S11 = L(span(0, pos - 1), span(0, pos - 1));
  mat S13 = L(span(0, pos - 1), span(pos + 1, L.n_cols - 1));
  mat S33 = L(span(pos + 1, L.n_rows - 1), span(pos + 1, L.n_cols - 1));
  mat S23 = L(pos, span(pos + 1, L.n_cols - 1));
  mat L_out(idx_old.n_elem - 1, idx_old.n_elem - 1, fill::zeros);
  L_out.submat(span(0, S11.n_rows - 1), span(0, S11.n_cols - 1)) = S11;
  L_out.submat(span(0, S11.n_rows - 1), span(S11.n_cols, L_out.n_cols - 1)) = S13;
  L_out.submat(span(S11.n_rows, L_out.n_rows - 1), 
               span(S11.n_cols, L_out.n_cols - 1)) = 
                 chol(S33.t() * S33 + S23.t() * S23);
  // find the output index for Cholesky decomposition
  uvec idx_out(idx_old.n_elem - 1);
  idx_out(span(0, pos - 1)) = idx_old(span(0, pos - 1));
  idx_out(span(pos, idx_out.n_elem - 1)) = idx_old(span(pos + 1, idx_old.n_elem - 1));
  // output
  field<mat> out(2);
  out(0) = L_out;
  out(1) = conv_to<mat>::from(idx_out);
  return out;
}

// [[Rcpp::export]]
field<mat> CholeskyUpdate(mat X, mat L, 
                          uvec idx_old, uvec idx_new) {
  // X - full symmetric matrix
  // L - upper triangle matrix of Cholesky decomposition for X[idx_old, idx_old]
  // idx_old - old index of L for the X[idx_old, idx_old]
  // idx_new - find the Cholesky decomposition for X[idx_new, idx_new]
  
  // first, we need to drop the rows
  uword drop_done = 0;
  while (drop_done == 0) {
    if (idx_old.is_empty()) {
      drop_done = 1;
    }
    for (uword i = 0; i < idx_old.n_elem; ++i) {
      uvec temp = find(idx_new == idx_old(i));
      if (temp.is_empty()) {
        field<mat> drop_temp = CholeskyDrop(X, L, idx_old, idx_new);
        L = drop_temp(0);
        idx_old = conv_to<uvec>::from(drop_temp(1));
        break;
      }
      if (i == idx_old.n_elem - 1) {
        drop_done = 1;
      }
    }
  }
  // second, add new rows
  field<mat> out = CholeskyAdd(X, L, idx_old, idx_new);
  return out;
}

// [[Rcpp::export]]
field<mat> ElasticNetALO_CholUpdate(vec beta, double intercept, 
                                    mat X, vec y, 
                                    double lambda, double alpha, 
                                    mat L_old, uvec idx_old) {
  // find out the dimension of X
  double n = X.n_rows;
  double p = X.n_cols;
  // define full data set
  vec ones(n,fill::ones);
  mat X_full = X;
  X_full.insert_cols(0, ones);
  // define full beta (intercept & slope)
  vec beta_full(p + 1);
  beta_full(0) = intercept;
  beta_full(span(1, p)) = beta;
  // compute prediction
  vec y_hat = X_full * beta_full;
  // find active set
  uvec A = find(beta_full != 0);
  // compute matrix H
  mat H(n, n, fill::zeros);
  uvec idx;
  mat L;
  if(!A.is_empty()) {
    mat X_active = X_full.cols(A);
    if (L_old.n_rows == 0) {
      L = chol(X_active.t() * X_active); // upper triangle matrix
      idx = A;
    } else {
      field<mat> update = CholeskyUpdate(X_full.t() * X_full, L_old, idx_old, A);
      L = update(0);
      idx = conv_to<uvec>::from(update(1));
    }
    mat L_t = L.t(); // take the transpose -> lower triangle matrix
    mat L_t_inv = inv(L_t);
    mat AE = L_t_inv * X_full(idx, idx).t();
    H = AE.t() * AE;
  }
  // compute the ALO prediction
  vec y_alo = y_hat + H.diag() % (y_hat - y) / (1 - H.diag());
  // generate the output variable
  field<mat> out(3);
  out(0) = conv_to<mat>::from(y_alo);
  out(1) = L;
  out(2) = conv_to<mat>::from(idx);
  return out;
}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
L = matrix(c(2, 0, 0, 6, 1, 0, -8, 5, 3), ncol = 3)
temp_add = CholeskyAdd(t(L)%*%L, L[c(1,2),c(1,2)], c(0,1), c(1,2,0))
temp_drop = CholeskyDrop(t(L)%*%L, L, c(0,1,2), c(0,1))
temp_update = CholeskyUpdate(t(L)%*%L, L[c(1,2),c(1,2)], c(0,1), c(1))
*/
