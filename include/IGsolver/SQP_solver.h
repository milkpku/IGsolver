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
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace IGsolver{
  namespace SQP{

    typedef Eigen::VectorXd               dVec;
    typedef Eigen::SparseMatrix<double>   SpMat;
    typedef SpMat::InnerIterator          Iter;

    typedef std::function<void(const dVec& X, double& eval, dVec& c)> Fun_eval;
    typedef std::function<void(const dVec& X, double& eval, dVec& grad, SpMat& hessian, dVec& c, SpMat& Jc)> Fun_grad_hess_Jc;
    typedef std::function<void(const int n_iter, const int cut_cnt, const dVec& X, const double& eval,
      const dVec& dX, const dVec& grad_res, const dVec& c)> Fun_iter;

    struct SQP_Config
    {
      int max_iter = 100;
      double g_res_norm = 1e-5;
      double dx_norm = 1e-5;
      double c_norm = 1e-5;
      int cut_iter = 10;
      double mu = 10;
    };

    void SQP_solver(dVec& solution, Fun_eval fun_eval, Fun_grad_hess_Jc fun_grad, Fun_iter iter_fun, SQP_Config config);

  }
}
