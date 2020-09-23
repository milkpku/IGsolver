# IGsolver

This library implements basic optimization algorithms, including [Newton's Method](https://en.wikipedia.org/wiki/Newton%27s_method) for unconstrained problems and [SQP method](https://en.wikipedia.org/wiki/Sequential_quadratic_programming) for constrained problems. It uses [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) as matrix wrapper and take modern C++ function object as input.

### Unconstrained problem
`IGsolver::Chol_solver` uses Newton's Method to solve unconstrained problems, and implements [LM method](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) to resolve singularity. You can customize the shifting matrix of LM method.

If the hessian of optimization problem is not positive definite, Cholesky decomposition may fail. To fix this problem, change `SOLVER` defined in `chol_solver.cpp` to sparse matrix solvers that support non-positive definite matrices, like LU solver or LDLT solver.  

### Constrained problem
`IGsolver::SQP_solver` uses Sequential Quadratic Programming (SQP) method to solve constrained problems. It takes [augmented Lagrangian](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method) as merit function, and LU solver to decomposite SQP matrix.

## Installation

```sh
  git clone https://github.com/milkpku/IGsolver
  cd IGsolver
  mkdir build
  cd build
  cmake ..
```

After compilation, you can get library `IGsolver.lib`, link it to your own project.

## Usage
### Chol solver

```c++
  #include <IGsolver/Chol_Solver.h> 
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
    printf("[iter %d] e = %g, grad = %g, dX = %g, cut_cnt = %d\n", 
      n_iter, eval, grad.norm(), dX.norm(), cut_cnt);
  };

  Chol_Config config;
  dVec sol(2);
  sol << 1, 0.7;
  Chol_solver(sol, f_eval, f_grad, f_iter, config);

```

### SQP solver
```c++
  #include <IGsolver/SQP_Solver.h> 
  using namespace IGsolver::SQP;

  /* min x1^2 + x2^2 
   * s.t.  x1^2 - x2 -1 = 0
   */
  Fun_eval func_eval = [](const dVec& X, double& eval, dVec& c)
  {
    eval = X.squaredNorm();
    c.resize(1);
    c(0) = X(0) * X(0) - X(1) - 1;
  };

  Fun_grad_hess_Jc func_grad_hess_Jc = [](const dVec& X, double& eval, 
      dVec& grad, SpMat& hessian, dVec& c, SpMat& Jc)
  {
    eval = X.squaredNorm();
    grad = 2 * X;
    hessian.resize(2, 2);
    hessian.insert(0, 0) = 2;
    hessian.insert(1, 1) = 2;

    c.resize(1);
    c(0) = X(0) * X(0) - X(1) - 1;
    Jc.resize(1, 2);
    Jc.insert(0, 0) = 2 * X(0);
    Jc.insert(0, 1) = -1;
  };

  Fun_iter func_iter = [](const int n_iter, const int cut_cnt, const dVec& X, 
    const double& eval, const dVec& dX, const dVec& grad_res, const dVec& c)
  {
    printf("[iter %d] (x1, x2) = (%g, %g), f(x) = %g, gnorm = %g, c = %g\n",
        n_iter, X(0), X(1), eval, grad_res.norm(), c(0));
  };

  SQP_Config config;

  dVec sol = dVec::Random(2);

  SQP_solver(sol, func_eval, func_grad_hess_Jc, func_iter, config);

```

## Advanced Usage

Eigen's sparse matrix solver is slow when matrix becomes big, so please consider other sparse matrix solvers like [PARDISO](https://www.pardiso-project.org/) or [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html). Fortunately, Eigen has convenient API to connect PARDISO and SuiteSparse. 

Just redefine `SOLVER` in `src/chol_solver.cpp` and `src/SQP_solver.cpp` as recommended in comment. For example, adding
```c++
 #include <Eigen/CholmodSupport>
 #define SOLVER Eigen::CholmodSupernodalLLT<SpMat>
```
 in the beginning of `src/Chol_solver.cpp` replaces Eigen's solver with SuiteSparse Cholesky solver, which speeds up the decomposition for several magnitudes. Also, adding
```c++
  #include <Eigen/UmfPackSupport>
  #define SOLVER Eigen::UmfPackLU<SpMat>
 ```
in the beginning of `src/SQP_solver.cpp` replaces Eigen's solver with SuiteSparse Umfpack LU solver. For more information of thrid-party sparse matrix solvers, please refer to Eigen's manual of [Solving Sparse System](https://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html).

You can customize config settings in `Chol_config` and `SQP_config`, which are self-documented.
