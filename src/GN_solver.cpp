/*
 * Copyright 2018 Li-Ke Ma <milkpku.gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 */
#include "IGsolver/GN_solver.h"

#include <ctime>
#include <iostream>

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
namespace GN
{
  bool GN_solver(dVec& X, Fun_eval fun_eval, Fun_res_jacobian fun_grad, 
      Fun_iter iter_fun, GN_Config config)
  {
    std::clock_t start;
    start = std::clock();

    /* initialization */
    dVec res;
    SpMat jacobian;

    fun_grad(X, res, jacobian);

    /* exame validity of e, grad and hess */
    if (!std::isfinite(res.sum())) return false;
    if (!std::isfinite(jacobian.sum())) return false;

    double e = res.squaredNorm();
    dVec grad = jacobian.transpose() * res;
    SpMat hess = jacobian.transpose() * jacobian;

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

    while ((grad.norm() > config.grad_norm || dX.norm() > config.dx_norm) 
        && n_iter < config.max_iter)
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
        if (!config.silent) 
          printf("fail to factorize W, try lambda = %g\n", lambda);
        hess += lambda * D;
        solver.factorize(hess);
      }

      if (valid_cnt >= config.valid_iter) return false;

      /* solve and cut */
      dX = -solver.solve(grad);
      if (dX.norm() > config.max_dx_norm)
        dX *= config.max_dx_norm / dX.norm();
      if (dX.dot(grad) > 0)
        dX *= -1;
      fun_eval(X, dX, res);
      double e_next = res.squaredNorm();

      int cut_cnt = 0;
      while ((std::isnan(e_next) || e_next > e) && cut_cnt < config.cut_iter 
          && valid_cnt < config.valid_iter)
      {
        if (std::isnan(e_next))
        {
          valid_cnt++;
          lambda *= 10;
          if (!config.silent) 
            printf("solution not valid, try lambda %g\n", lambda);
        }
        else {
          cut_cnt++;
          lambda *= 2;
          if (!config.silent) 
            printf("energy not decrease, %g > %g, try lambda = %g\n", 
                e_next, e, lambda);
        }
        hess += lambda * D;
        solver.factorize(hess);
        dX = -solver.solve(grad);
        fun_eval(X, dX, res);
        e_next = res.squaredNorm();
      }

      if (valid_cnt >= config.valid_iter) return false;

      /* step forward */
      X += dX;
      fun_grad(X, res, jacobian);
      e = res.squaredNorm();
      grad = jacobian.transpose() * res;
      hess = jacobian.transpose() * jacobian;

      /* output */
      iter_fun(n_iter, cut_cnt, X, e, dX, grad);
    }

    if (!(grad.norm() > config.grad_norm || dX.norm() > config.dx_norm))
    {
      double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
      if (!config.silent) printf("Finish Solving, takes %gs\n", duration);
      return true;
    }
    else
    {
      double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
      if (!config.silent) printf("Fail Solving, takes %gs\n", duration);
      return false;
    }
  }
}
}
