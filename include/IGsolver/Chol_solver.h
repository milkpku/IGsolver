#include <functional>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace IGsolver
{
  typedef Eigen::VectorXd               dVec;
  typedef Eigen::SparseMatrix<double>   SpMat;
  typedef SpMat::InnerIterator          Iter;
  typedef Eigen::Triplet<double>        Triplet;

  typedef std::function<void(const dVec& X, double& eval, dVec& grad, SpMat& hessian)> Fun_grad_hessian;
  typedef std::function<bool(const dVec& X, const dVec& dX, const double e, double& e_next)> Fun_valid_decrease;
  typedef std::function<void(const int n_iter, const int cut_cnt, const dVec& X, const double& eval,
    const dVec& dX, const dVec& grad)> Fun_iter;


  struct Chol_Config
  {
    int max_iter = 200;
    double grad_norm = 1e-5;
    double dx_norm = 1e-5;
    int valid_iter = 40;
    int cut_iter = 10;
    bool custom_damping = false;
    SpMat D;
    bool silent = false;
  };

  bool Chol_solver(dVec& solution, Fun_grad_hessian fun_eval, Fun_valid_decrease fun_valid, Fun_iter iter_fun, Chol_Config config);
}
