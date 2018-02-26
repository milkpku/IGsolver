#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace IGsolver{
  namespace SQP{

    typedef Eigen::VectorXd               dVec;
    typedef Eigen::SparseMatrix<double>   SpMat;
    typedef SpMat::InnerIterator          Iter;

    typedef std::function<void(const dVec& X, double& eval, dVec& grad, SpMat& hessian)> Fun_grad_hessian;
    typedef std::function<void(const dVec& X, dVec& c, SpMat& Jc)> Fun_constraint;
    typedef std::function<void(const int n_iter, const int cut_cnt, const dVec& X, const double& eval,
      const dVec& dX, const dVec& grad_res, const dVec& c)> Fun_iter;

    struct SQP_Config
    {
      double g_res_norm = 1e-5;
      double dx_norm = 1e-5;
      double c_norm = 1e-5;
      int cut_iter = 10;
    };

    void SQP_solver(dVec& solution, Fun_grad_hessian fun_eval, Fun_constraint fun_cons, Fun_iter iter_fun, SQP_Config config);

  }
}
