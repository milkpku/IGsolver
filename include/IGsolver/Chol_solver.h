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
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace IGsolver{
  namespace Chol{

    typedef Eigen::VectorXd               dVec;
    typedef Eigen::SparseMatrix<double>   SpMat;
    typedef SpMat::InnerIterator          Iter;

    typedef std::function<void(const dVec& X, double& eval, dVec& grad, SpMat& hessian)> Fun_grad_hessian;
    typedef std::function<void(const dVec& X, const dVec& dX, const double e, double& e_next)> Fun_eval;
    typedef std::function<void(const int n_iter, const int cut_cnt, const dVec& X, const double& eval,
      const dVec& dX, const dVec& grad)> Fun_iter;


    struct Chol_Config
    {
      int max_iter = 200;
      double grad_norm = 1e-5;
      double dx_norm = 1e-5;
      int valid_iter = 40;
      int cut_iter = 10;
      bool custom_damping = false;
      SpMat D;
      bool silent = false;
    };

    bool Chol_solver(dVec& solution, Fun_eval fun_eval, Fun_grad_hessian fun_grad, Fun_iter iter_fun, Chol_Config config);
  
  }
}
