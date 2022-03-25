#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

double sign(double a) {
  double mysign;
  if(a == 0) {mysign=0;} else {mysign = (a>0 ? 1 : -1);}
  return mysign;
}

double absf(double a) {
  double value = (a>0 ? a : ((-1)*a));
  return value;
}


double neg_loglik_cpp(arma::mat& X, arma::colvec& time, arma::colvec& delta,
                   arma::colvec& beta) {
  int nrow = X.n_rows;
  
  // initialization
  double loglik = 0;
  arma::mat at_risk(nrow,1);
  arma::mat xbeta = X*beta;
  arma::mat exp_xbeta = exp(xbeta);
  double time_i;
  
  // calculation
  loglik += arma::as_scalar(xbeta.t()*delta);
  for(int i=0; i<nrow; i++) {
    time_i = time(i);
    if(delta(i) != 0) {
      for(int k=0; k<nrow; k++) {
        at_risk(k,0) = (time(k) >= time_i ? 1 : 0);
      }
      loglik += (-1.0)*log(arma::as_scalar(exp_xbeta.t()*at_risk)/(double)nrow);
    }
  }
  
  return 0 - 1.0/(double)nrow*loglik;
}


void neg_loglik_functions_cpp(double& neg_loglik, arma::colvec& neg_dloglik, arma::mat& neg_ddloglik, 
                          arma::mat& X, arma::colvec& time, arma::colvec& delta, arma::colvec& beta) {
  int ncol = X.n_cols;
  int nrow = X.n_rows;
  
  // initialization
  arma::mat xbeta = X*beta;
  arma::mat exp_xbeta = exp(xbeta);
  double mu0_i;
  arma::colvec mu1_i(ncol);
  arma::mat mu2_i(ncol, ncol);
  arma::mat at_risk(nrow, 1);
  double time_i;
  neg_loglik = 0;
  neg_dloglik.fill(0);
  neg_ddloglik.fill(0);
  
  // calculation
  neg_loglik += arma::as_scalar(xbeta.t()*delta);
  for(int i=0; i<nrow; i++) {
    time_i = time(i);
    if(delta(i) != 0) {
      for(int k=0; k<nrow; k++) {
        at_risk(k,0) = (time(k) >= time_i ? 1 : 0);
      }
      mu0_i = arma::as_scalar(exp_xbeta.t()*at_risk)/(double)nrow;
      mu1_i = trans(X)*(at_risk%exp_xbeta)/(double)nrow;
      mu2_i = trans(X)*diagmat(vectorise(at_risk%exp_xbeta))*X/(double)nrow;
      neg_loglik += 0 - log(mu0_i);
      neg_dloglik += trans(X.row(i)) - mu1_i/mu0_i;
      neg_ddloglik += mu2_i/mu0_i - (mu1_i/mu0_i)*trans(mu1_i/mu0_i);
    }
  }
  
  neg_loglik = (0.0-1.0)/((double)nrow)*neg_loglik;
  neg_dloglik = (0.0-1.0)/((double)nrow)*neg_dloglik;
  neg_ddloglik = neg_ddloglik/((double)nrow);
} 


arma::colvec lasso_univCox_cpp(double lambda, arma::mat& X, arma::colvec& time, 
							 arma::colvec& delta, arma::colvec& beta_init, 
							 double tol=1.0e-6, int maxiter=1000) {
  int iter=1;
  double beta_diff=100000.0;  
  int ncol = X.n_cols;
  arma::colvec beta_old = beta_init;
  arma::colvec beta_new = beta_init;
  double neg_loglik; 
  arma::colvec neg_dloglik(ncol);
  arma::mat neg_ddloglik(ncol, ncol);
  double a_d;
  
  while((beta_diff>tol) & (iter<=maxiter)) {
    neg_loglik = 0;
    neg_dloglik.fill(0);
    neg_ddloglik.fill(0);
    neg_loglik_functions_cpp(neg_loglik, neg_dloglik, neg_ddloglik, X, time, delta, beta_old);
    for(int d=0; d<ncol; d++) {
      a_d = arma::as_scalar(trans(beta_new)*neg_ddloglik.col(d)) - beta_new(d)*neg_ddloglik(d,d) -
        arma::as_scalar(trans(beta_old)*neg_ddloglik.col(d)) + neg_dloglik(d);
      beta_new(d) = (0.0-1)*sign(a_d)*(absf(a_d)>lambda ? (absf(a_d) - lambda) : 0)/neg_ddloglik(d,d);
    }
    
    beta_diff = sqrt(sum(pow(beta_new-beta_old, 2)));
    iter = iter + 1;
    beta_old = beta_new;
  }// end while
  
  return beta_new;
} 


//[[Rcpp::export]]
void neg_loglik_functions_cpp_ext(double& neg_loglik, arma::colvec& neg_dloglik, arma::mat& neg_ddloglik, 
                              arma::mat& score_sq, arma::mat& X, arma::colvec& time, arma::colvec& delta, arma::colvec& beta) {
  int ncol = X.n_cols;
  int nrow = X.n_rows;
  
  // initialization
  arma::mat xbeta = X*beta;
  arma::mat exp_xbeta = exp(xbeta);
  double mu0_i;
  arma::colvec mu1_i(ncol);
  arma::mat mu2_i(ncol, ncol);
  arma::mat at_risk(nrow, 1);
  double time_i;
  neg_loglik = 0;
  neg_dloglik.fill(0);
  neg_ddloglik.fill(0);
  score_sq.fill(0);
  
  // calculation
  neg_loglik += arma::as_scalar(xbeta.t()*delta);
  for(int i=0; i<nrow; i++) {
    time_i = time(i);
    if(delta(i) != 0) {
      for(int k=0; k<nrow; k++) {
        at_risk(k,0) = (time(k) >= time_i ? 1 : 0);
      }
      mu0_i = arma::as_scalar(exp_xbeta.t()*at_risk)/(double)nrow;
      mu1_i = trans(X)*(at_risk%exp_xbeta)/(double)nrow;
      mu2_i = trans(X)*diagmat(vectorise(at_risk%exp_xbeta))*X/(double)nrow;
      neg_loglik += 0 - log(mu0_i);
      neg_dloglik += trans(X.row(i)) - mu1_i/mu0_i;
      neg_ddloglik += mu2_i/mu0_i - (mu1_i/mu0_i)*trans(mu1_i/mu0_i);
      score_sq += (trans(X.row(i)) - mu1_i/mu0_i)*(X.row(i) - trans(mu1_i/mu0_i));
    }
  }
  
  neg_loglik = (0.0-1.0)/((double)nrow)*neg_loglik;
  neg_dloglik = (0.0-1.0)/((double)nrow)*neg_dloglik;
  neg_ddloglik = neg_ddloglik/((double)nrow);
  score_sq = score_sq/((double)nrow);
} 


// [[Rcpp::export]]
arma::colvec lasso_univCox_cpp_ext(double lambda, arma::mat& X, arma::colvec& time, 
                               arma::colvec& delta, arma::colvec& beta_init, 
                               double tol=1.0e-6, int maxiter=1000) {
  int iter=1;
  double beta_diff=100000.0;  
  int ncol = X.n_cols;
  arma::colvec beta_old = beta_init;
  arma::colvec beta_new = beta_init;
  double neg_loglik; 
  arma::colvec neg_dloglik(ncol);
  arma::mat neg_ddloglik(ncol, ncol);
  double a_d;
  
  while((beta_diff>tol) & (iter<=maxiter)) {
    neg_loglik = 0;
    neg_dloglik.fill(0);
    neg_ddloglik.fill(0);
    neg_loglik_functions_cpp(neg_loglik, neg_dloglik, neg_ddloglik, X, time, delta, beta_old);
    for(int d=0; d<ncol; d++) {
      a_d = arma::as_scalar(trans(beta_new)*neg_ddloglik.col(d)) - beta_new(d)*neg_ddloglik(d,d) -
        arma::as_scalar(trans(beta_old)*neg_ddloglik.col(d)) + neg_dloglik(d);
      beta_new(d) = (0.0-1)*sign(a_d)*(absf(a_d)>lambda ? (absf(a_d) - lambda) : 0)/neg_ddloglik(d,d);
    }
    
    beta_diff = sqrt(sum(pow(beta_new-beta_old, 2)));
    iter = iter + 1;
    beta_old = beta_new;
  }// end while
  
  return beta_new;
} 


// [[Rcpp::export]]
List cv_lasso_univCox_cpp(arma::mat& X, arma::colvec& time, arma::colvec& delta, 
                        arma::colvec& beta_init, arma::uvec& cv_idx, int nfold=5, 
                        int nlambda=50, double tol=1.0e-6, int maxiter=1000) {
  // initialization
  int ncol = X.n_cols;
  int nrow = X.n_rows;
  arma::colvec zeros_p = arma::zeros<arma::colvec>(ncol);
  arma::vec lambda_seq(nlambda);
  double lambda_max = 0.0;
  double lam;
  double lambda_min_ratio = (nrow > ncol ? 0.0001 : 0.01);
  arma::vec cv_value = arma::zeros<arma::vec>(nlambda);
  arma::colvec beta_tmp(beta_init);
  
  arma::uvec keep_idx = find(cv_idx != 1);
  int n_keep = keep_idx.n_elem;
  arma::mat X_tmp(n_keep, ncol);
  arma::colvec time_tmp(n_keep), delta_tmp(n_keep);
  
  double neg_loglik=0;
  arma::colvec neg_dloglik(ncol);
  neg_dloglik.fill(0);
  arma::mat neg_ddloglik(ncol, ncol);
  neg_ddloglik.fill(0);
  double a_d;
  
  arma::colvec beta_opt(ncol);
  double lambda_opt;
  
  // compute lambda sequence 
  neg_loglik_functions_cpp(neg_loglik, neg_dloglik, neg_ddloglik, X, time, delta, zeros_p);
  for(int d=0; d<ncol; d++) {
    a_d = neg_dloglik(d);
    if(absf(a_d) > lambda_max) lambda_max = absf(a_d);
  }
  for(int m=0; m<nlambda; m++) {
    lambda_seq(m) = lambda_max*pow(lambda_min_ratio, (double)m/(double)(nlambda-1));
  }
  
  // cross-validation to get cv_value
  for(int m=0; m<nlambda; m++) {
    lam = lambda_seq(m);
    for(int j=1; j<=nfold; j++) {
      keep_idx = find(cv_idx != j);
      X_tmp = X.rows(keep_idx);
      time_tmp = time.elem(keep_idx);
      delta_tmp = delta.elem(keep_idx);
      
      // calculate beta_tmp with data withheld
      beta_tmp = lasso_univCox_cpp(lam, X_tmp, time_tmp, delta_tmp, beta_tmp, tol, maxiter);
      cv_value(m) += ((double)nrow)*(0.0-1)*neg_loglik_cpp(X, time, delta, beta_tmp) -
        ((double)n_keep)*(0.0-1)*neg_loglik_cpp(X_tmp, time_tmp, delta_tmp, beta_tmp);
    } // end for (nfold)
    Rcout << "Cross-validation with Lambda No." << (m+1) << " is done!"<< std::endl;
  } // end for (nlambda)
  
  lambda_opt = lambda_seq(0);
  double tmp = cv_value(0);
  for(int i=1; i<nlambda; i++) {
    if(cv_value(i) > tmp) {
      tmp = cv_value(i);
      lambda_opt = lambda_seq(i);
    }
  }
  
  beta_opt = lasso_univCox_cpp(lambda_opt, X, time, delta, beta_tmp, tol, maxiter);
  neg_loglik=0;
  neg_dloglik.fill(0);
  neg_ddloglik.fill(0);
  neg_loglik_functions_cpp(neg_loglik, neg_dloglik, neg_ddloglik, X, time, delta, beta_opt);
  
  return List::create(_["beta_opt"] = beta_opt, _["lambda_seq"] = lambda_seq, 
                      _["cv_value"] = cv_value, _["lambda_opt"] = lambda_opt,
                      _["neg_loglik"] = neg_loglik, _["neg_dloglik"] = neg_dloglik, _["neg_ddloglik"] = neg_ddloglik);
}


//[[Rcpp::export]]
arma::mat C_mat(arma::mat& X, arma::colvec& time, arma::colvec& delta, arma::colvec& beta) {
  int nrow = X.n_rows;
  int ncol = X.n_cols;
  arma::mat C(nrow*nrow, ncol); C.fill(0);
  arma::mat at_risk(nrow, 1);
  arma::mat xbeta = X*beta;
  arma::mat exp_xbeta = exp(xbeta);
  double time_i;
  double mu0_i = 0.0;
  arma::colvec mu1_i(ncol); mu1_i.fill(0);
  arma::mat tmp(nrow, ncol);
  
  for(int i=0; i<nrow; i++) {
    time_i = time(i);
    for(int k=0; k<nrow; k++) {
      at_risk(k,0) = (time(k) >= time_i ? 1 : 0);
    }
    mu0_i = arma::as_scalar(trans(at_risk)*exp_xbeta)/(double)nrow;
    mu1_i = trans(X)*(at_risk%exp_xbeta)/(double)nrow;
    tmp = X.each_row() - trans(mu1_i/mu0_i);
    C.rows(i*nrow, (i+1)*nrow-1) = delta(i)*(tmp.each_col()%(at_risk%sqrt(exp_xbeta/mu0_i)))/(double)nrow;
  }// end for i
  
  return C;
}


// [[Rcpp::export]]
double C_stat_harrell(arma::mat& Z, arma::colvec& beta, arma::colvec& delta, arma::colvec& time) {
  int n = Z.n_rows;
  //int p = Z.n_cols;
  arma::colvec zbeta = Z*beta;
  double denom = 0.0, num = 0.0;
  for(int i=0; i<n; i++) {
    if( delta(i) == 1.0 ) {
      for(int j=0; j<n; j++) {
        if( j != i) {
          num += (((time(i) < time(j)) & (zbeta(i) > zbeta(j))) ? 1.0 : 0.0);
          denom += (time(i) < time(j) ? 1.0 : 0.0);
        } // end if in j
      } // end for j
    } // end if in i
  } // end for i
  
  double Cstat = num / denom;
  return Cstat;
} // function to compute Harrell's C-statistic


// [[Rcpp::export]]
double loglik_cpp_ext(arma::mat& X, arma::colvec& time, arma::colvec& delta,
                      arma::colvec& beta) {
  int nrow = X.n_rows;
  
  // initialization
  double loglik = 0;
  arma::mat at_risk(nrow,1);
  arma::mat xbeta = X*beta;
  arma::mat exp_xbeta = exp(xbeta);
  double time_i;
  
  // calculation
  loglik += arma::as_scalar(xbeta.t()*delta);
  for(int i=0; i<nrow; i++) {
    time_i = time(i);
    if(delta(i) != 0) {
      for(int k=0; k<nrow; k++) {
        at_risk(k,0) = (time(k) >= time_i ? 1 : 0);
      }
      loglik += (-1.0)*log(arma::as_scalar(exp_xbeta.t()*at_risk)/(double)nrow);
    }
  }
  
  return 1.0/(double)nrow*loglik;
}