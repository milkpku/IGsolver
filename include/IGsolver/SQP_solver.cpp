#include "SQP_solver.h"
#include <milkLib/SuiteSparseLib.h>
#include <Eigen/UmfPackSupport>
#include <Eigen/CholmodSupport>
#include <vector>
#include <iostream>
#include <boost/format.hpp>

namespace IGsolver
{
  void SQP_solver(dVec& solution, Fun_grad_hessian fun_eval, Fun_constraint fun_cons, Iter_fun iter_fun, SQP_Config config)
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
    Eigen::UmfPackLU<SpMat> solver;
    solver.analyzePattern(W);
    double c_norm = c.norm();

    /* starg loop */
    int n_iter = 0;
    dVec dX = dVec::Ones(solution.size());
    dVec grad_res = dVec::Ones(c.size());
    dVec dX_mean = dX;
    dVec dX_var = dX;
    double ossilation_decay = 1.f / config.ossilation_window;
    bool ossilate = false;
    while ((grad_res.norm() > config.g_res_norm || dX.norm() > config.dx_norm) && !ossilate)
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
        while(e_next > e && c.norm() > c_norm && cut_cnt < config.cut_iter)
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

      dX_mean *= 1 - ossilation_decay;
      dX_mean += dX * ossilation_decay;
      dX_var *= 1 - ossilation_decay;
      dX_var += dX.cwiseAbs2() * ossilation_decay;
      if (n_iter > 2 * config.ossilation_window)
      {
        int p_max_var;
        dX_var.maxCoeff(&p_max_var);
        double sig = abs(dX_mean(p_max_var)) / sqrt(dX_var(p_max_var));
        if (sig < config.ossilation_threshold) { ossilate = true; };
        std::cout << "ossilation watcher: " << sig << "\n";
      }

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
