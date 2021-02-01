#include <functional>
#include <iostream>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "IGsolver/GN_solver.h"

int main(int argc, char* argv[])
{
  using namespace IGsolver::GN;
  /* f = (1.5 - x + xy)^2 + (2.25 -x + xy^2)^2 + (2.625 - x + xy^3)^2
   * res = [1.5 - x + xy; 2.25 -x + xy^2; 2.625 - x + xy^3] 
   * jacobian = [y-1, x; y^2-1, 2xy; y^3-1, 3xy^2]
   *
   * solution is f(3, 0.5) = 0
   */
  Fun_res_jacobian f_grad = [](const dVec& X, dVec& res, 
      SpMat& jacobian)
  {
    double x = X(0);
    double y = X(1);

    res.resize(3);
    res(0) = 1.5 - x + x*y;
    res(1) = 2.25 - x + x*y*y;
    res(2) = 2.625 - x + x*y*y*y;

    jacobian.resize(3, 2);
    jacobian.insert(0, 0) = y-1;
    jacobian.insert(0, 1) = x;
    jacobian.insert(1, 0) = y*y-1;
    jacobian.insert(1, 1) = 2*x*y;
    jacobian.insert(2, 0) = y*y*y-1;
    jacobian.insert(2, 1) = 3*x*y*y;
  };

  Fun_eval f_eval = [](const dVec& X, const dVec& dX, dVec& res)
  {
    double x = X(0);
    double y = X(1);

    res.resize(3);
    res(0) = 1.5 - x + x*y;
    res(1) = 2.25 - x + x*y*y;
    res(2) = 2.625 - x + x*y*y*y;
  };

  Fun_iter f_iter = [](const int n_iter, const int cut_cnt, const dVec& X, 
      const double& eval, const dVec& dX, const dVec& grad)
  {
    printf("[iter %d] e = %g, grad = %g, dX = %g, cut_cnt = %d\n", 
      n_iter, eval, grad.norm(), dX.norm(), cut_cnt);
  };

  GN_Config config;
  dVec sol(2);
  sol << 0, 0;
  GN_solver(sol, f_eval, f_grad, f_iter, config);

  printf("solution f(%g, %g) is minimal\n", sol(0), sol(1));

  system("pause");
  return 0;
}
