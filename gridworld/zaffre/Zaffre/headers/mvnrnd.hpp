#ifndef _ZAFFRE_MVNRND_H_
#define _ZAFFRE_MVNRND_H_

#include <Zaffre/headers/Zaffre.hpp>

/**
See matlab mvnrnd - multivariate normal distribution sampling.
Code from: http://www.boost.org/doc/libs/1_57_0/doc/html/boost/variate_generator.html
*/
class mvnrnd {
public:
	mvnrnd(const VectorXd & meanVec, const MatrixXd & covarMat, const int & RNGSeed);
	~mvnrnd();
	void setCovar(const MatrixXd & covarMat);
	void setMean(const VectorXd & meanVec);
	void sample(VectorXd & buff);

	// If you will only sample once, or call mvnpdf once, then use these. If you will call many times, make the object so you don't recompute eigenvectors.
	static void sample(VectorXd & buff, VectorXd & meanVec, const MatrixXd & covarMat, const int & RNGSeed);

private:
	int n;	// Dimension
	mt19937_64 generator;
	normal_distribution<double> * norm; // standard normal distribution
	MatrixXd rot;	// [n][n]
	VectorXd scl;	// [n][1]
	VectorXd mean;	// [n][1]
};

void test_mvnrnd();

#endif
