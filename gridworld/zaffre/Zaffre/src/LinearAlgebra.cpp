#include <Zaffre/headers/Zaffre.hpp>

/* Returns true iff m is positive semidefinite. Checks the eigenvalues are greater than -epsilon (so epislon should be a small positive constant) */
bool positiveSemidefinite(const MatrixXd & m, double epsilon)
{
	return (m.eigenvalues().real().array() > -epsilon).all();
}

/* Returns true iff m is positive definite. Checks that all of the eigenvalues are greater than epsilon (a small positive constant) */
bool positiveDefinite(const MatrixXd & m, double epsilon)
{
	return (m.eigenvalues().real().array() > epsilon).all();
}

/* Find least-squares solution, solves Ax=b (returns the least-squares x). A = matrix, b = vector, result = vector */
VectorXd linsolve(const MatrixXd & A, const VectorXd & b)
{
	JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
	return svd.solve(b);
}

// A possibly more-effficient implementation (might avoid a copy, depending on how smart the compiler is)
MatrixXd pinv(const MatrixXd & a, double epsilon)
{
	if (a.rows() < a.cols())
	{
		cerr << "Error: pinv only takes matrices with more (or equal) columns than rows." << endl;
		exit(1);
	}
	Eigen::JacobiSVD<MatrixXd> svd = a.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
	double tolerance = epsilon * max((double)a.cols(), (double)a.rows()) * svd.singularValues().array().abs().maxCoeff();
	return svd.matrixV() * MatrixXd((svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0)).asDiagonal() * svd.matrixU().adjoint();
}
