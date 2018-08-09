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

// generate a Givens rotation matrix
// a, b - vector to be rotated
//[[Rcpp::export]]
mat givensRotate(const double &a, const double &b) {
  mat G(2, 2);
  
  double c, s, tau;
  if (b == 0) {
    c = 1;
    s = 0;
  } else if (abs(b) > abs(a)) {
    tau = -a / b;
    s = 1 / sqrt(1 + tau * tau);
    c = s * tau;
  } else {
    tau = -b / a;
    c = 1 / sqrt(1 + tau * tau);
    s = c * tau;
  }
  G(0, 0) = c;
  G(0, 1) = -s;
  G(1, 0) = s;
  G(1, 1) = c;
  
  return -G;
}

//[[Rcpp::export]]
mat cholDowndate(const mat &R, uvec drop) {
  mat S = R;
  uword dropId;
  for (uword i = 0; i < drop.n_elem; i++) {
    dropId = drop(i);
    S.shed_col(dropId);
    if (dropId != R.n_cols - 1) {
      for (uword j = dropId; j < S.n_cols; j++) {
        S.submat(j, j, j + 1, S.n_cols - 1) = givensRotate(S(j, j), S(j + 1, j)) * S.submat(j, j, j + 1, S.n_cols - 1);
      }
    }
    S.shed_row(S.n_rows - 1);
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
  uvec dropIdx;
  mat Q;
  for (uword i = 1; i < beta.n_cols; i++) {
    // checkUserInterrupt();
    toAdd = addList(i - 1) - 1; // columns to add
    toDrop = dropList(i - 1) - 1; // columns to drop
    
     //auto start2 = std::chrono::steady_clock::now();
    if(toDrop.n_elem > 0) {
      dropIdx.set_size(toDrop.n_elem);
      for(uword j = 0; j < toDrop.n_elem; j++) {
        dropIdx(j) = conv_to<uword>::from(find(currXIdx == toDrop(j), 1, "first"));
        currXIdx.shed_row(dropIdx(j));
      }
      XE = X.cols(currXIdx);
      // qr_econ(Q, R, XE);
      // R = cholDowndate(R, dropIdx);
      R = chol(XE.t() * XE);
    }
    //auto end2 = std::chrono::steady_clock::now();
    //Rcout << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms for Drop" << std::endl;
    //auto start = std::chrono::steady_clock::now();
    if(toAdd.n_elem > 0) {
      R = cholUpdate(R, XE, X.cols(toAdd));
      currXIdx = join_cols(currXIdx, toAdd);
      XE = X.cols(currXIdx);
    }
    //auto end = std::chrono::steady_clock::now();
    //Rcout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms for Add" << std::endl;
    // compute updates of y
    theta = y - X * beta.col(i);
    XER = solve(trimatl(R.t()), XE.t());
    aloUpdate.col(i) = theta / diagvec(In - XER.t() * XER);
  }
  
  return aloUpdate;
}

//[[Rcpp::export]]
mat lassoALOWoodbury(const mat &X, const vec &y, const sp_mat &beta, 
                 const field<uvec> &activeList, const field<uvec> &addList, const field<uvec> &dropList) {
  uvec currXIdx = activeList(0) - 1;
  mat aloUpdate(X.n_rows, beta.n_cols, fill::zeros); // matrix storing updates of y
  vec theta = y - X * beta.col(0);
  mat In = eye<mat>(X.n_rows, X.n_rows);
  mat XE = X.cols(currXIdx);
  mat invXTX = inv_sympd(XE.t() * XE);
  mat J = XE * invXTX * XE.t();
  aloUpdate.col(0) = theta / diagvec(In - J);
  
  // update through all lambdas
  uvec toAdd;
  uvec toDrop;
  uvec dropIdx;
  uvec XEcolIdx;
  uvec invPerm;
  mat Xdrop;
  mat Ainv, B, D;
  mat AinvB, E, AinvBE;
  mat T11, T12, T22;
  for (uword i = 1; i < beta.n_cols; i++) {
    // checkUserInterrupt();
    toAdd = addList(i - 1) - 1; // columns to add
    toDrop = dropList(i - 1) - 1; // columns to drop
    
    //auto start2 = std::chrono::steady_clock::now();
    if(toDrop.n_elem > 0) {
      XEcolIdx = regspace<uvec>(0, currXIdx.n_elem - 1);
      dropIdx.set_size(toDrop.n_elem);
      for(uword j = 0; j < toDrop.n_elem; j++) {
        dropIdx(j) = conv_to<uword>::from(find(currXIdx == toDrop(j), 1, "first"));
        currXIdx.shed_row(dropIdx(j));
        XEcolIdx.shed_row(dropIdx(j));
      }
      invPerm = join_cols(XEcolIdx, dropIdx);
      invXTX = invXTX.submat(invPerm, invPerm);
      XE = X.cols(currXIdx);
      T11 = invXTX.submat(0, 0, XEcolIdx.n_elem - 1, XEcolIdx.n_elem - 1);
      T12 = invXTX.submat(0, XEcolIdx.n_elem, XEcolIdx.n_elem - 1, invXTX.n_cols - 1);
      T22 = invXTX.submat(XEcolIdx.n_elem, XEcolIdx.n_elem, invXTX.n_rows - 1, invXTX.n_cols - 1);
      invXTX = T11 - T12 * solve(T22, T12.t()); // use solve?
      // invXTX = inv(XE.t() * XE);
    }
    //auto end2 = std::chrono::steady_clock::now();
    //Rcout << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms for Drop" << std::endl;
    //auto start = std::chrono::steady_clock::now();
    if(toAdd.n_elem > 0) {
      currXIdx = join_cols(currXIdx, toAdd);
      Ainv = invXTX;
      B = XE.t() * X.cols(toAdd);
      D = trans(X.cols(toAdd)) * X.cols(toAdd);
      XE = X.cols(currXIdx);
      
      AinvB = Ainv * B;
      E = inv(D - B.t() * invXTX * B);
      AinvBE = AinvB * E;
      
      invXTX.set_size(currXIdx.n_elem, currXIdx.n_elem);
      invXTX.submat(0, 0, Ainv.n_rows - 1, Ainv.n_cols - 1) = Ainv + AinvBE * AinvB.t();
      invXTX.submat(0, Ainv.n_cols, Ainv.n_rows - 1, invXTX.n_cols - 1) = -AinvBE;
      invXTX.submat(Ainv.n_rows, 0, invXTX.n_rows - 1, Ainv.n_cols - 1) = -AinvBE.t();
      invXTX.submat(Ainv.n_rows, Ainv.n_cols, invXTX.n_rows - 1, invXTX.n_cols - 1) = E;
    }
    //auto end = std::chrono::steady_clock::now();
    //Rcout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms for Add" << std::endl;
    // compute updates of y
    theta = y - X * beta.col(i);
    J = XE * invXTX * XE.t();
    aloUpdate.col(i) = theta / diagvec(In - J);
  }
  
  return aloUpdate;
}