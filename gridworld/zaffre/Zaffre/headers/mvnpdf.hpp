#ifndef _ZAFFRE_MVNPDF_H_
#define _ZAFFRE_MVNPDF_H_

#include <Zaffre/headers/Zaffre.hpp>

/*
See matlab mvnpdf
*/
class mvnpdf {
public:
	mvnpdf(const VectorXd & meanVec, const MatrixXd & covarMat);
	void setCovar(const MatrixXd & covarMat);
	void setMean(const VectorXd & meanVec);
	double Pr(const VectorXd & x); // Get probability of the specified x

	// If you will only sample once, or call mvnpdf once, then use these. If you will call many times, make the object so you don't recompute eigenvectors.
	static double Pr(const VectorXd & x, VectorXd & meanVec, const MatrixXd & covarMat);

private:
	int n;	// Dimension
	VectorXd mu;
	MatrixXd Sigma;
	MatrixXd SigmaInv;
	double DetermSigma;
};

void test_mvnpdf();

#endif
