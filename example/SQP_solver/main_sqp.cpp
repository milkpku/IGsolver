#include <iostream>
#include <boost/format.hpp>

#include "SQP_solver.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>



int main(int argc, char* argv[])
{
  using namespace IGsolver::SQP;

  /* min x1^2 + x2^2 
   * s.t.  x1^2 - x2 -1 = 0
   */
  Fun_eval func_eval = [](const dVec& X, double& eval, dVec& c)
  {
    eval = X.squaredNorm();
    c.resize(1);
    c(0) = X(0) * X(0) - X(1) - 1;
  };

  Fun_grad_hess_Jc func_grad_hess_Jc = [](const dVec& X, double& eval, 
      dVec& grad, SpMat& hessian, dVec& c, SpMat& Jc)
  {
    eval = X.squaredNorm();
    grad = 2 * X;
    hessian.resize(2, 2);
    hessian.insert(0, 0) = 2;
    hessian.insert(1, 1) = 2;

    c.resize(1);
    c(0) = X(0) * X(0) - X(1) - 1;
    Jc.resize(1, 2);
    Jc.insert(0, 0) = 2 * X(0);
    Jc.insert(0, 1) = -1;
  };

  Fun_iter func_iter = [](const int n_iter, const int cut_cnt, const dVec& X, 
    const double& eval, const dVec& dX, const dVec& grad_res, const dVec& c)
  {
    std::cout << boost::format("[iter %d] (x1, x2) = (%g, %g), f(x) = %g, gnorm = %g, c = %g\n")
      % n_iter % X(0) % X(1) % eval % grad_res.norm() % c(0);
  };

  SQP_Config config;

  dVec sol = dVec::Random(2);

  SQP_solver(sol, func_eval, func_grad_hess_Jc, func_iter, config);

  system("pause");
}
