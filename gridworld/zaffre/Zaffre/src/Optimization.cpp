#include <Zaffre/headers/Zaffre.hpp>

// Return a vector containing the indices in sorted order
template <typename T>
vector<int> sort_indexes(const vector<T> &v) {
	vector<int> idx(v.size());
	for (int i = 0; i < (int)idx.size(); ++i) idx[i] = i;							// initialize original index locations
	sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2]; });	// sort indexes based on comparing values in v
	return idx;
}

VectorXd CMAES(const VectorXd & initialMean,	// The initial mean
	const double & initialSigma,				// The initial standard deviation (multiplied by a predetermined covariance matrix)
	int numIterations,							// Number of iterations to run before stopping
	double(*f)(const VectorXd &, void *[]),		// The function to be optimized. The first parameter is the point, the second contains any other necessary information.
	void * data[],								// Additional data to be sent to f whenever called
	bool minimize,								// Minimize or maximize?
	int RNGSeed,
	int lambda) { // Default value = -1 for lambda - set in CMAES.h

	int N = (int)initialMean.size(), hsig;
	if (lambda == -1)
		lambda = 4 + (int)floor(3.0 * log(N));
	double sigma = initialSigma, mu = lambda / 2.0, eigeneval = 0, chiN = pow(N, 0.5)*(1.0 - 1.0 / (4.0*N) + 1.0 / (21.0*N*N));;					// number of parents/points for recombination
	VectorXd xmean = initialMean, weights((int)mu);					// Initial center
	for (int i = 0; i < (int)mu; i++)
		weights[i] = i + 1;
	weights = log(mu + 1.0 / 2.0) - weights.array().log();				// muXone array for weighted recombination
	mu = floor(mu);
	weights = weights / weights.sum();									// normalize recombination weights array
	double mueff = weights.sum()*weights.sum() / weights.dot(weights),
		cc = (4.0 + mueff / N) / (N + 4.0 + 2.0*mueff / N),				// time constant for cumulation for C
		cs = (mueff + 2.0) / (N + mueff + 5.0),							// t-const for cumulation for sigma control
		c1 = 2.0 / ((N + 1.3)*(N + 1.3) + mueff),						// learning rate for rank-one update of c
		cmu = min(1.0 - c1, 2.0*(mueff - 2.0 + 1.0 / mueff) / ((N + 2.0)*(N + 2.0) + mueff)),	// and for rank-mu update.
		damps = 1.0 + 2.0*max(0.0, sqrt((mueff - 1.0) / (N + 1.0)) - 1.0) + cs;					// damping for sigma, usually close to 1.
	VectorXd pc = VectorXd::Zero(N), ps = VectorXd::Zero(N), D = VectorXd::Ones(N), DSquared = D, DInv = 1.0 / D.array(), xold, oneOverD;
	for (int i = 0; i < (int)DSquared.size(); i++)
		DSquared[i] *= DSquared[i];
	MatrixXd B = MatrixXd::Identity(N, N), C = B * DSquared.asDiagonal() * B.transpose(), invsqrtC = B * DInv.asDiagonal() * B.transpose(), arx(N, lambda), arxSubMatrix(N, (int)(mu + .1)),	// arx(:,arindex(1:mu)) ---- Holds the columns from the best mu entries according to fitness.
		repmat(xmean.size(), (int)(mu + .1)), artmp;

	vector<double> arfitness(lambda);
	vector<int> arindex;	// Used later. Declare outside this loop so that we can return values from it.
	mt19937_64 generator(RNGSeed); // If you put this inside the loops, you get bad behavior. Don't!

	vector<mt19937_64> threadGenerators;
	for (int counteval = 0; counteval < numIterations;) {
		for (int k = 0; k < lambda; k++) {
			// Load the k'th column of arx with a new point
			normal_distribution<double> distribution(0, 1);
			VectorXd randomVector(N);
			for (int i = 0; i < N; i++)
				randomVector[i] = D[i] * distribution(generator);
			arx.col(k) = xmean + sigma * B * randomVector; // Random vector includes the multiplication (element wise) by D.
			arfitness[k] = (minimize ? 1 : -1)*f(arx.col(k), data);	// Compute the new point's fitness and store it
		}

		counteval += lambda;	// Update counteval
		//cout << "counteval = " << counteval << endl;
		arindex = sort_indexes(arfitness);
		xold = xmean;
		for (int col = 0; col < mu; col++)
			arxSubMatrix.col(col) = arx.col(arindex[col]);
		xmean = arxSubMatrix*weights;
		ps = (1.0 - cs)*ps + sqrt(cs*(2.0 - cs)*mueff) * invsqrtC * (xmean - xold) / sigma;
		hsig = (ps.norm() / sqrt(1.0 - pow(1.0 - cs, 2.0 * counteval / lambda)) / (double)chiN < 1.4 + 2.0 / (N + 1.0) ? 1 : 0);
		pc = (1 - cc)*pc + hsig * sqrt(cc*(2 - cc)*mueff) * (xmean - xold) / sigma;
		for (int i = 0; i < repmat.cols(); i++)
			repmat.col(i) = xold;
		artmp = (1.0 / sigma) * (arxSubMatrix - repmat);
		C = (1 - c1 - cmu) * C + c1 * (pc*pc.transpose() + (1 - hsig) * cc*(2 - cc) * C) + cmu * artmp * weights.asDiagonal() * artmp.transpose();
		sigma = sigma * exp((cs / damps)*(ps.norm() / (double)chiN - 1.0));
		if (counteval - eigeneval > lambda / (c1 + cmu) / (double)N / 10.0) {
			eigeneval = counteval;
			for (int r = 0; r < C.rows(); r++) {
				for (int c = r + 1; c < C.cols(); c++)
					C(r, c) = C(c, r);
			}
			EigenSolver<MatrixXd> es(C);	// Eigen solver for eigenvectors
			D = C.eigenvalues().real();
			B = es.eigenvectors().real();
			D = D.array().sqrt();
			for (int i = 0; i < B.cols(); i++)
				B.col(i) = B.col(i).normalized();
			oneOverD = 1.0 / D.array();
			invsqrtC = B * oneOverD.asDiagonal() * B.transpose();
		}
	}

	return arx.col(arindex[0]);
}

double objective_function_test_CMAES(const VectorXd & x, void * data[]) {
	return x.dot(x);
}

void testCMAES() {
	VectorXd initialMean(20);
	initialMean.setConstant(1);
	double initialSigma = 1;
	int numIterations = 5000;
	VectorXd cmaesResult = CMAES(initialMean, initialSigma, numIterations, objective_function_test_CMAES, NULL, true, true, rand());

	cout << "CMA-ES found minimum of x.dot(x) at = " << cmaesResult.transpose() << endl;
	cout << "Press enter to continue." << endl;
	forceGetchar();
}
