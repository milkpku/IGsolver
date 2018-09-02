#include <functional>
#include <iostream>
#include <cmath>

#include <boost/format.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "IGsolver/Chol_solver.h"


int main(int argc, char* argv[])
{
  using namespace IGsolver::Chol;
  /* f = 0.5 * x1^2 * (x_1^2/6 + 1) + x2 * arctan(x2) - 0.5 * ln(x2^2 + 1)
   * f' = [x1^3/3 + x1; arctan(x2)]
   * f'' = diag{x1^2 + 1, 1/(1+x2^2)}
   */
  Fun_grad_hessian f_grad = [](const dVec& X, double& eval, dVec& grad, SpMat& hessian)
  {
    double x1 = X(0);
    double x2 = X(1);

    eval = 0.5 * x1 * x1 * (x1 * x1 / 6 + 1) + x2 * atan(x2) - 0.5 * log(x2 * x2 + 1);

    grad.resize(2);
    grad(0) = x1 * x1 * x1 / 3 + x1;
    grad(1) = atan(x2);

    hessian.resize(2, 2);
    hessian.insert(0, 0) = x1 * x1 + 1;
    hessian.insert(1, 1) = 1 / (1 + x2 * x2);
  };

  Fun_eval f_eval = [](const dVec& X, const dVec& dX, const double e, double& e_next)
  {
    double x1 = X(0) + dX(0);
    double x2 = X(1) + dX(1);

    e_next = 0.5 * x1 * x1 * (x1 * x1 / 6 + 1) + x2 * atan(x2) - 0.5 * log(x2 * x2 + 1);
  };

  Fun_iter f_iter = [](const int n_iter, const int cut_cnt, const dVec& X, const double& eval, const dVec& dX, const dVec& grad)
  {
    std::cout << boost::format("[iter %d] e = %g, grad = %g, dX = %g, cut_cnt = %d\n") 
      % n_iter % eval % grad.norm() % dX.norm() % cut_cnt;
  };

  Chol_Config config;
  dVec sol(2);
  sol << 1, 0.7;
  Chol_solver(sol, f_eval, f_grad, f_iter, config);

  system("pause");
  return 0;
}
