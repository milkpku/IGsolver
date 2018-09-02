#include "IGsolver/Chol_solver.h"

#include <ctime>
#include <iostream>
#include <boost/format.hpp>

/*  You can replace SOLVER with any other Cholesky solvers, such as
*  
*  #include <Eigen/CholmodSupport>
*  #pragma comment(lib, ${SuiteSparse_Lib})
*  #pragma comment(lib, ${MKL_Lib})
*  #define SOLVER Eigen::CholmodSupernodalLLT<SpMat>
*  
*/

#ifndef SOLVER
#define SOLVER Eigen::SimplicialCholesky<SpMat>
#endif

namespace IGsolver {
namespace Chol
{
  bool Chol_solver(dVec& X, Fun_eval fun_eval, Fun_grad_hessian fun_grad, Fun_iter iter_fun, Chol_Config config)
  {
    std::clock_t start;
    start = std::clock();

    /* initialization */
    double e;
    dVec grad;
    SpMat hess;

    fun_grad(X, e, grad, hess);

    /* exame validity of e, grad and hess */
    if (!isfinite(e)) return false;
    if (!isfinite(grad.sum())) return false;
    if (!isfinite(hess.sum())) return false;

    /* prepare solver */
    SOLVER solver;

    /* prepare damping */
    SpMat D;
    if (config.custom_damping) { D = config.D; }
    else { D.resize(hess.innerSize(), hess.outerSize()); D.setIdentity(); };

    int n_iter = 0;
    dVec dX = dVec::Zero(X.size());

    double lambda = 1e-3 * abs(hess.diagonal().mean());
    hess += lambda * D;
    solver.analyzePattern(hess);

    while ((grad.norm() > config.grad_norm || dX.norm() > config.dx_norm) && n_iter < config.max_iter)
    {
      n_iter++;

      lambda /= 10;
      hess += lambda * D;

      /* factorize hessian */
      int valid_cnt = 0;
      solver.factorize(hess);
      if (solver.info() != Eigen::Success && valid_cnt < config.valid_iter)
      {
        valid_cnt++;
        lambda *= 10;
        if (!config.silent) std::cout << boost::format("fail to factorize W, try lambda = %g\n") % lambda;
        hess += lambda * D;
        solver.factorize(hess);
      }

      if (valid_cnt >= config.valid_iter) return false;

      /* solve and cut */
      dX = -solver.solve(grad);
      double e_next;
      fun_eval(X, dX, e, e_next);
      int cut_cnt = 0;
      while ((isnan(e_next) || e_next > e) && cut_cnt < config.cut_iter && valid_cnt < config.valid_iter)
      {
        if (isnan(e_next))
        {
          valid_cnt++;
          lambda *= 10;
          if (!config.silent) std::cout << boost::format("solution not valid, try lambda %g\n") % lambda;
        }
        else {
          cut_cnt++;
          lambda *= 2;
          if (!config.silent) std::cout << boost::format("energy not decrease, %g > %g, try lambda = %g\n") % e_next % e % lambda;
        }
        hess += lambda * D;
        solver.factorize(hess);
        dX = -solver.solve(grad);
        fun_eval(X, dX, e, e_next);
      }

      if (valid_cnt >= config.valid_iter) return false;

      /* step forward */
      X += dX;
      fun_grad(X, e, grad, hess);

      /* output */
      iter_fun(n_iter, cut_cnt, X, e, dX, grad);
    }

    if (!(grad.norm() > config.grad_norm || dX.norm() > config.dx_norm))
    {
      double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
      if (!config.silent) std::cout << boost::format("Finish Solving, takes %gs\n") % duration;
      return true;
    }
    else
    {
      double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
      if (!config.silent) std::cout << boost::format("Fail Solving, takes %gs\n") % duration;
      return false;
    }
  }
}
}
