#ifndef _OPTIMIZATION_H_
#define _OPTIMIZATION_H_

#include <Zaffre/headers/Zaffre.hpp>

//////////////////////////////////////////////////
///// Function prototypes
//////////////////////////////////////////////////

// Implements CMA-ES (http://en.wikipedia.org/wiki/CMA-ES). Return value is the minimizer / maximizer
VectorXd CMAES(const VectorXd & initialMean,	// The initial mean
	const double & initialSigma,				// The initial standard deviation (multiplied by a predetermined covariance matrix)
	int numIterations,							// Number of iterations to run before stopping
	double(*f)(const VectorXd &, void *[]),		// The function to be optimized. The first parameter is the point, the second contains any other necessary information.
	void * data[],								// Additional data to be sent to f whenever called
	bool minimize,								// Minimize or maximize?
	int RNGSeed,
	int lambda = -1);							// Population size. Will be auto-set if you don't set it

void testCMAES();								// Simple test of CMAES

#endif
