#include "SQP_solver.h"

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
  void SQP_solver(dVec& solution, Fun_grad_hessian fun_eval, Fun_constraint fun_cons, Fun_iter iter_fun, SQP_Config config)
  {
    /* initialization */
    double e;
    dVec grad;
    SpMat hessian;
    dVec c;
    SpMat Jc;

    fun_eval(solution, e, grad, hessian);
    fun_cons(solution, c, Jc);

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

    /* preparing loop */
    SpMat W = get_W(hessian, Jc);
    dVec res = get_res(grad, c);
    SOLVER solver;
    solver.analyzePattern(W);
    double c_norm = c.norm();

    /* starg loop */
    int n_iter = 0;
    dVec dX = dVec::Ones(solution.size());
    dVec grad_res = dVec::Ones(c.size());
    dVec dX_mean = dX;
    dVec dX_var = dX;
    while ((grad_res.norm() > config.g_res_norm || dX.norm() > config.dx_norm))
    {
      n_iter++;
      /* solve problem */
      solver.factorize(W);
      if (solver.info() != Eigen::Success)
      {
        std::cout << "fail to fractorize W\n";
        assert(false && "numerical error");
        return;
      }
      dVec dL = solver.solve(res);

      /* test constraint satisfication */
      int cut_cnt = 0;
      fun_cons(solution + dL.head(solution.size()), c, Jc);
      while (c.norm() > std::max(c_norm, config.c_norm) && cut_cnt < config.cut_iter)
      {
        std::cout << boost::format("[Constriant cut]: c_next > max(c, config), %g > max(%g, %g)\n")
          % c.norm() % c_norm % config.c_norm;
        cut_cnt++;
        dL /= 2;
        fun_cons(solution + dL.head(solution.size()), c, Jc);
      }

      /* if SQP is satisfied, test energy decrease */
      double e_next;
      fun_eval(solution + dL.head(solution.size()), e_next, grad, hessian);
      if (cut_cnt == 0)
      {
        while (e_next > e && c.norm() > c_norm && cut_cnt < config.cut_iter)
        {
          std::cout << boost::format("[Energy cut]: e_next > e , %g > %g, c_next > c, %g > %g\n") % e_next % e % c.norm() % c_norm;
          cut_cnt++;
          dL /= 2;
          fun_eval(solution + dL.head(solution.size()), e_next, grad, hessian);
          fun_cons(solution + dL.head(solution.size()), c, Jc);
        }
      }

      /* update */
      dX = dL.head(solution.size());
      solution += dX;

      e = e_next;
      c_norm = c.norm();

      /* residual */
      grad_res = grad + Jc.transpose() * dL.tail(c.size());

      iter_fun(n_iter, cut_cnt, solution, e, dX, grad_res, c);

      W = get_W(hessian, Jc);
      res = get_res(grad, c);
    }

    /* finish solving */
    std::cout << boost::format("Finish Solving\n");
  }
}
}