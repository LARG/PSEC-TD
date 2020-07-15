#ifndef _ZAFFRE_LINEARALGEBRA_H_
#define _ZAFFRE_LINEARALGEBRA_H_

#include <Zaffre/headers/Zaffre.hpp>

/* Returns true iff m is positive semidefinite. Checks the eigenvalues */
bool positiveSemidefinite(const MatrixXd & m, double epsilon = 0.0000001);

/* Returns true iff m is positive definite. Checks the eigenvalues */
bool positiveDefinite(const MatrixXd & m, double epsilon = 0.0000001);

/* Find least-squares solution, solves Ax=b (returns the least-squares x). A = matrix, b = vector, result = vector */
VectorXd linsolve(const MatrixXd & A, const VectorXd & b);

// Compute Moore-penrose pseudoinverse
MatrixXd pinv(const MatrixXd & a, double epsilon = std::numeric_limits<double>::epsilon());

#endif
