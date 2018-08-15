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
mat lassoALOChol(const mat &X, const vec &y, const sp_mat &beta, 
                 const field<uvec> &activeList, const field<uvec> &addList, const field<uvec> &dropList) {
  uvec currXIdx = activeList(0);
  mat aloUpdate(X.n_rows, beta.n_cols, fill::zeros); // matrix storing updates of y
  vec theta = y - X * beta.col(0);
  mat In = eye<mat>(X.n_rows, X.n_rows);
  mat XE = X.cols(currXIdx);
  mat R = chol(XE.t() * XE);
  mat XER = solve(trimatl(R.t()), XE.t());
  vec J = sum(square(XER.t()), 1);
  aloUpdate.col(0) = theta / (1 - J);

  // update through all lambdas
  uvec toAdd;
  uvec toDrop;
  uvec dropIdx;
  uvec XEcolIdx;
  mat Q;
  for (uword i = 1; i < beta.n_cols; i++) {
    // checkUserInterrupt();
    toAdd = addList(i - 1); // columns to add
    toDrop = dropList(i - 1); // columns to drop
    
    if(toDrop.n_elem > 0) {
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
      R = chol(XE.t() * XE);
    }
    if(toAdd.n_elem > 0) {
      R = cholUpdate(R, XE, X.cols(toAdd));
      currXIdx = join_cols(currXIdx, toAdd);
      XE = X.cols(currXIdx);
    }
    // compute updates of y
    theta = y - X * beta.col(i);
    XER = solve(trimatl(R.t()), XE.t());
    J = sum(square(XER.t()), 1);
    aloUpdate.col(i) = theta / (1 - J);
  }
  
  return aloUpdate;
}

void milUpdate(const uvec &toAdd, const mat &B, const mat &D, mat &invMat, uvec &currID) {
  currID = join_cols(currID, toAdd);
  
  mat AinvB = invMat * B;
  mat E = inv_sympd(D - B.t() * AinvB); // Schur complement
  mat AinvBE = AinvB * E;
  mat AinvPlus = invMat + AinvBE * AinvB.t();
  invMat.set_size(currID.n_elem, currID.n_elem);
  invMat.submat(0, 0, AinvPlus.n_rows - 1, AinvPlus.n_cols - 1) = AinvPlus;
  invMat.submat(0, AinvPlus.n_cols, AinvPlus.n_rows - 1, currID.n_elem - 1) = -AinvBE;
  invMat.submat(AinvPlus.n_rows, 0, currID.n_elem - 1, AinvPlus.n_cols - 1) = -AinvBE.t();
  invMat.submat(AinvPlus.n_rows, AinvPlus.n_cols, currID.n_elem - 1, currID.n_elem - 1) = E;
}

void milDowndate(const uvec &toDrop, mat &invMat, uvec &XEIdx, uvec &currID) {
  uvec dropIdx;
  dropIdx.zeros(toDrop.n_elem);
  for (uword j = 0; j < toDrop.n_elem; j++) {
    dropIdx(j) = conv_to<uword>::from(find(currID == toDrop(j), 1, "first"));
  }
  dropIdx = sort(dropIdx, "descend");
  for (uword j = 0; j < toDrop.n_elem; j++) {
    currID.shed_row(dropIdx(j));
    XEIdx.shed_row(dropIdx(j));
  }
  uvec invPerm = join_cols(XEIdx, dropIdx);
  
  invMat = invMat.submat(invPerm, invPerm);
  mat T11 = invMat.submat(0, 0, XEIdx.n_elem - 1, XEIdx.n_elem - 1);
  mat T12 = invMat.submat(0, XEIdx.n_elem, XEIdx.n_elem - 1, invMat.n_cols - 1);
  mat T22 = invMat.submat(XEIdx.n_elem, XEIdx.n_elem, invMat.n_rows - 1, invMat.n_cols - 1);
  invMat = T11 - T12 * solve(T22, T12.t());
}

//[[Rcpp::export]]
// Update matrix inverse through matrix inversion lemma
mat lassoALOMIL(const mat &X, const vec &y, const sp_mat &beta, 
                 const field<uvec> &activeList, const field<uvec> &addList, const field<uvec> &dropList) {
  uvec currXIdx = activeList(0);
  mat aloUpdate(X.n_rows, beta.n_cols, fill::zeros); // matrix storing updates of y
  vec theta = y - X * beta.col(0);
  mat XTX = X.t() * X;
  mat XE = X.cols(currXIdx);
  mat invXTX = inv_sympd(XE.t() * XE);
  vec J = sum((XE * invXTX) % XE, 1);
  aloUpdate.col(0) = theta / (1 - J);
  
  // update through all lambdas
  uvec toAdd;
  uvec toDrop;
  uvec XEcolIdx;
  for (uword i = 1; i < beta.n_cols; i++) {
    // checkUserInterrupt();
    toAdd = addList(i - 1); // columns to add
    toDrop = dropList(i - 1); // columns to drop
    
    if (currXIdx.n_elem > 0) {
      // remove variables
      if (toDrop.n_elem > 0) {
        XEcolIdx = regspace<uvec>(0, currXIdx.n_elem - 1);
        milDowndate(toDrop, invXTX, XEcolIdx, currXIdx);
        XE = XE.cols(XEcolIdx);
      }
      // add variables
      if (toAdd.n_elem > 0) {
        milUpdate(toAdd, XTX.submat(currXIdx, toAdd), XTX.submat(toAdd, toAdd), invXTX, currXIdx);
        XE = join_rows(XE, X.cols(toAdd));
      }
    } else {
      currXIdx = toAdd;
      XE = X.cols(toAdd);
      invXTX = inv_sympd(XE.t() * XE);
    }
    
    // compute ALO
    theta = y - X * beta.col(i);
    J = sum((XE * invXTX) % XE, 1);
    aloUpdate.col(i) = theta / (1 - J);
  }
  
  return aloUpdate;
}

//[[Rcpp::export]]
// Update matrix inverse through matrix inversion lemma
mat elnetALOApprox(const mat &X, const vec &y, const sp_mat &beta, const vec &lambda, const double &alpha,
                   const field<uvec> &activeList, const field<uvec> &addList, const field<uvec> &dropList) {
  uvec currXIdx = activeList(0);
  mat aloUpdate(X.n_rows, beta.n_cols, fill::zeros); // matrix storing updates of y
  vec yHat = X * beta.col(0);
  mat XE = X.cols(currXIdx);
  mat IS = eye<mat>(XE.n_cols, XE.n_cols);
  mat A = XE.t() * XE / X.n_rows + (1 - alpha) * lambda(0) * IS;
  mat F = inv_sympd(A);
  vec H = sum((XE * F) % XE, 1);
  aloUpdate.col(0) = yHat + H % (yHat - y) / (X.n_rows - H);
  
  // update through all lambdas
  uvec toAdd;
  uvec toDrop;
  uvec XEcolIdx;
  uvec dropIdx;
  mat AF;
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
        IS = eye<mat>(XE.n_cols, XE.n_cols);
        A = XE.t() * XE / X.n_rows + (1 - alpha) * lambda(i) * IS;
      }
    } else {
      currXIdx = toAdd;
      XE = X.cols(toAdd);
      F = inv_sympd(XE.t() * XE);
    }
    // approx inverse
    IS = eye<mat>(XE.n_cols, XE.n_cols);
    XE = X.cols(activeList(i));
    A = XE.t() * XE / X.n_rows + (1 - alpha) * lambda(i) * IS;
    F = A.t() / std::pow(norm(A, "fro"), 2);
    AF = A * F;
    F = F * (3 * IS - AF * (3 * IS - AF)); // cubic iteration
    
    // compute ALO
    yHat = X * beta.col(i);
    H = sum((XE * F) % XE, 1);
    aloUpdate.col(i) = yHat + H % (yHat - y) / (X.n_rows - H);
  }
  
  return aloUpdate;
}