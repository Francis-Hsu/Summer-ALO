#include <RcppArmadillo.h>
#include <math.h>

using namespace Rcpp;
using namespace arma;

//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

//[[Rcpp::export]]
mat cholUpdate(const mat &R, const mat &X, const mat &A) {
  mat S12 = solve(trimatl(R.t()), X.t() * A);
  mat S(R.n_rows + A.n_cols, R.n_cols + A.n_cols, fill::zeros);
  S.submat(0, 0, R.n_rows - 1, R.n_cols - 1) = R;
  S.submat(0, R.n_cols, R.n_rows - 1, S.n_cols - 1) = S12;
  S.submat(R.n_rows, R.n_cols, S.n_rows - 1, S.n_cols - 1) = chol(A.t() * A - S12.t() * S12);
  
  return S;
}

//[[Rcpp::export]]
mat cholDowndateOne(const mat &R, uword i) {
  mat S(R.n_rows - 1, R.n_cols - 1, fill::zeros);
  if(i == R.n_cols - 1) {
    S = R.submat(0, 0, R.n_rows - 2, R.n_cols - 2);
  } else {
    mat S23 = R.submat(i, i + 1, i, R.n_cols - 1);
    mat S33 = R.submat(i + 1, i + 1, R.n_rows - 1, R.n_cols - 1);
    if (i == 0) {
      S = chol(S33.t() * S33 + S23.t() * S23);
    } else {
      S.submat(0, 0, i - 1, i - 1) = R.submat(0, 0, i - 1, i - 1);
      S.submat(0, i, i - 1, S.n_cols - 1) = R.submat(0, i + 1, i - 1, R.n_cols - 1);
      S.submat(i, i, S.n_rows - 1, S.n_cols - 1) = chol(S33.t() * S33 + S23.t() * S23); 
    }
  }
  
  return S;
}

//[[Rcpp::export]]
mat lassoALOChol(const mat &X, const vec &y, const sp_mat &beta, 
                 const field<uvec> &activeList, const field<uvec> &addList, const field<uvec> &dropList) {
  uvec currXIdx = activeList(0) - 1;
  mat aloUpdate(X.n_rows, beta.n_cols, fill::zeros); // matrix storing updates of y
  vec theta = y - X * beta.col(0);
  mat In = eye<mat>(X.n_rows, X.n_rows);
  mat XE = X.cols(currXIdx);
  mat R = chol(XE.t() * XE);
  mat XER = solve(trimatl(R.t()), XE.t());
  aloUpdate.col(0) = theta / diagvec(In - XER.t() * XER);

  // update through all lambdas
  uvec toAdd;
  uvec toDrop;
  for (uword i = 1; i < beta.n_cols; i++) {
    // checkUserInterrupt();
    toAdd = addList(i - 1) - 1; // columns to add
    toDrop = dropList(i - 1) - 1; // columns to drop
    
    // auto start2 = std::chrono::steady_clock::now();
    if(toDrop.n_elem > 0) {
      for(uword j = 0; j < toDrop.n_elem; j++) {
        uword dropIdx = conv_to<uword>::from(find(currXIdx == toDrop(j), 1, "first"));
        currXIdx.shed_row(dropIdx);
        // R = cholDowndateOne(R, dropIdx);
      }
      XE = X.cols(currXIdx);
      R = chol(XE.t() * XE);
    }
    // auto end2 = std::chrono::steady_clock::now();
    // Rcout << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms for Drop" << std::endl;
    // auto start = std::chrono::steady_clock::now();
    if(toAdd.n_elem > 0) {
      R = cholUpdate(R, XE, X.cols(toAdd));
      currXIdx = join_cols(currXIdx, toAdd);
      XE = X.cols(currXIdx);
    }
    // auto end = std::chrono::steady_clock::now();
    // Rcout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms for Add" << std::endl;
    // compute updates of y
    theta = y - X * beta.col(i);
    XER = solve(trimatl(R.t()), XE.t());
    aloUpdate.col(i) = theta / diagvec(In - XER.t() * XER);
  }
  
  return aloUpdate;
}