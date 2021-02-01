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
  namespace GN{

    typedef Eigen::VectorXd               dVec;
    typedef Eigen::SparseMatrix<double>   SpMat;
    typedef SpMat::InnerIterator          Iter;

    typedef std::function<void(const dVec& X, dVec& res, SpMat& jacobian)> Fun_res_jacobian;
    typedef std::function<void(const dVec& X, const dVec& dX, dVec& res)> Fun_eval;
    typedef std::function<void(const int n_iter, const int cut_cnt, const dVec& X, 
        const double& eval, const dVec& dX, const dVec& grad)> Fun_iter;

    struct GN_Config
    {
      int max_iter = 200;
      double grad_norm = 1e-5;
      double dx_norm = 1e-5;
      int valid_iter = 40;
      int cut_iter = 10;
      bool custom_damping = false;
      SpMat D;
      double max_dx_norm = 1e6; 
      bool silent = false;
    };

    bool GN_solver(dVec& solution, Fun_eval fun_eval, Fun_res_jacobian fun_grad, Fun_iter iter_fun, GN_Config config);
  
  }
}
