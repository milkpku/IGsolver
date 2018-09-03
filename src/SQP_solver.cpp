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
#include "IGsolver/SQP_solver.h"

#include <vector>
#include <iostream>
#include <boost/format.hpp>

/*  You can replace SOLVER with any other LU solvers, such as
*
*  #include <Eigen/UmfPackSupport>
*  #pragma comment(lib, ${SuiteSparse_Lib})
*  #pragma comment(lib, ${MKL_Lib})
*  #define SOLVER Eigen::UmfPackLU<SpMat>
*
*/

#ifndef SOLVER
#define SOLVER Eigen::SparseLU<SpMat>
#endif


namespace IGsolver {
namespace SQP
{
  void SQP_solver(dVec& solution, Fun_eval fun_eval,  Fun_grad_hess_Jc fun_grad, Fun_iter iter_fun, SQP_Config config)
  {
    /* initialization */
    double e;
    dVec grad;
    SpMat hessian;
    dVec c;
    SpMat Jc;

    fun_grad(solution, e, grad, hessian, c, Jc);

    /* build SQP matrix and residual */
    typedef Eigen::Triplet<double> Triplet;
    auto get_W = [](const SpMat& H, const SpMat& J)
    {
      int num_var = H.outerSize();
      int num_c = J.innerSize();

      SpMat W(num_var + num_c, num_var + num_c);
      std::vector<Triplet> W_coeff;
      W_coeff.reserve(W.nonZeros() + 2 * J.nonZeros());
      for (int i = 0; i < H.outerSize(); i++)
        for (Iter it(H, i); it; ++it)
          W_coeff.push_back(Triplet(it.row(), it.col(), it.value()));

      for (int i = 0; i < J.outerSize(); i++)
        for (Iter it(J, i); it; ++it)
        {
          W_coeff.push_back(Triplet(it.row() + num_var, it.col(), it.value()));
          W_coeff.push_back(Triplet(it.col(), it.row() + num_var, it.value()));
        }

      W.setFromTriplets(W_coeff.begin(), W_coeff.end());
      return W;
    };

    auto get_res = [](const dVec& grad, const dVec& c)
    {
      dVec res(grad.size() + c.size());
      res << -grad, -c;
      return res;
    };

    auto merit_function = [&](const dVec& sol, const dVec& dX, const dVec& lambda)
    {
      double e_merit;
      dVec c_merit;
      fun_eval(sol + dX, e_merit, c_merit);
      e_merit += c_merit.dot(lambda) + config.mu * c_merit.squaredNorm();
      return e_merit;
    };

    /* preparing loop */
    SpMat W = get_W(hessian, Jc);
    dVec res = get_res(grad, c);
    SOLVER solver;
    solver.analyzePattern(W);

    /* starg loop */
    int n_iter = 0;
    double dx_norm = 1;
    dVec grad_res = dVec::Ones(c.size());
    while ((grad_res.norm() > config.g_res_norm || dx_norm > config.dx_norm))
    {
      n_iter++;
      /* solve problem */
      solver.factorize(W);
      if (solver.info() != Eigen::Success)
      {
        std::cout << "recompute W\n";
        solver.compute(W);
      }
      if (solver.info() != Eigen::Success)
      {
        std::cout << "fail to fractorize W\n";
        {
          // assert hessian nan
          for (int i = 0; i < hessian.outerSize(); i++)
            for (SpMat::InnerIterator it(hessian, i); it; ++it)
            {
              if (isnan(it.value()))
              {
                std::cout << boost::format("NaN in hessian : (%d, %d)\n") % it.row() % it.col();
              }
            }

          // assert Jc nan
          for (int i = 0; i < Jc.outerSize(); i++)
            for (SpMat::InnerIterator it(Jc, i); it; ++it)
            {
              if (isnan(it.value()))
              {
                std::cout << boost::format("NaN in Jc : (%d, %d)\n") % it.row() % it.col();
              }
            }
        }
        assert(false && "numerical error");
        return;
      }
      dVec dL = solver.solve(res);
      dVec dX = dL.head(solution.size());
      dVec lambda = dL.tail(c.size());

      /* test decrease of merit function
       * e_merit = e + lambda * c + mu * c^2
       *   */
      int cut_cnt = 0;
      double e_start = e + lambda.dot(c) + config.mu * c.squaredNorm();
      double e_next = merit_function(solution, dX, lambda);
      while ((isnan(e_next) || e_next > e_start) && cut_cnt < config.cut_iter)
      {
        std::cout << boost::format("[Energy cut]: e_next > e , %g > %g\n") % e_next % e_start;
        if (!isnan(e_next)) cut_cnt++;
        dX /= 2;
        e_next = merit_function(solution, dX, lambda);
      }

      /* update */
      solution += dX;
      dx_norm = dX.norm();

      fun_grad(solution, e, grad, hessian, c, Jc);
      W = get_W(hessian, Jc);
      res = get_res(grad, c);

      /* residual */
      grad_res = grad + Jc.transpose() * lambda;

      iter_fun(n_iter, cut_cnt, solution, e, dX, grad_res, c);
    }

    /* finish solving */
    std::cout << boost::format("Finish Solving\n");
  }
}
}
