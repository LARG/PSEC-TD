#include <Zaffre/headers/Zaffre.hpp>

mvnrnd::mvnrnd(const VectorXd & meanVec, const MatrixXd & covarMat, const int & RNGSeed) {
	generator.seed(RNGSeed);
	norm = new normal_distribution<double>(0, 1);
	n = (int)meanVec.size();
	setCovar(covarMat);
	setMean(meanVec);
}

mvnrnd::~mvnrnd() {
	delete norm;
}

void mvnrnd::setCovar(const MatrixXd & covarMat) {
	assert((covarMat.rows() == n) && (covarMat.cols() == n));	// If not, then make a new object.
	SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covarMat);
	rot = eigenSolver.eigenvectors();
	scl = eigenSolver.eigenvalues();
	for (int i = 0; i<n; ++i)
		scl(i, 0) = sqrt(scl(i, 0));
}

void mvnrnd::setMean(const VectorXd & meanVec) {
	assert((int)meanVec.size() == n); // If not, make a new object
	mean = meanVec;
}

void mvnrnd::sample(VectorXd & buff) {
	buff.resize(n);
	for (int i = 0; i < n; i++)
		buff(i, 0) = (*norm)(generator)*scl(i, 0);
	buff = rot*buff + mean;
}

void mvnrnd::sample(VectorXd & buff, VectorXd & meanVec, const MatrixXd & covarMat, const int & RNGSeed) {
	mt19937_64 generator(RNGSeed);
	normal_distribution<double> norm(0, 1);
	int n = (int)meanVec.size();

	// Overwrite private vars
	MatrixXd rot;	// [n][n]
	VectorXd scl;	// [n][1]
	VectorXd mean;	// [n][1]

	// Set covariance matrix
	assert((covarMat.rows() == n) && (covarMat.cols() == n));	// If not, then make a new object.
	SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covarMat);
	rot = eigenSolver.eigenvectors();
	scl = eigenSolver.eigenvalues();
	for (int i = 0; i<n; ++i)
		scl(i, 0) = sqrt(scl(i, 0));

	// Set mean
	assert((int)meanVec.size() == n); // If not, make a new object
	mean = meanVec;

	// Sample
	buff.resize(n);
	for (int i = 0; i < n; i++)
		buff(i, 0) = norm(generator)*scl(i, 0);
	buff = rot*buff + mean;
}

void test_mvnrnd() {
	cout << "Starting test of mvnrnd. Plots should all be similar [middle plot is overlay of other two]." << endl;
	for (int i = 0; i < 3; i++) {
		VectorXd mu(2);
		MatrixXd Sigma(2, 2);
		mt19937_64 gen((int)time(NULL));
		mu[0] = rand(gen, -10.0, 10.0);
		mu[1] = rand(gen, -10.0, 10.0);

		// Lower Triangular
		Sigma(0, 0) = rand(gen, 0.01, 10);// Diagonal must be positive
		Sigma(0, 1) = 0;
		Sigma(1, 0) = rand(gen, -10, 10);
		Sigma(1, 1) = rand(gen, 0.01, 10); // Diagonal must be positive

		Sigma = Sigma*Sigma.transpose();	// Make it a PSD matrix

		cout << "mu = " << mu.transpose() << endl;
		cout << "Sigma = " << endl << Sigma << endl;

		mvnrnd generator(mu, Sigma, (int)time(NULL));
		int n = intRand(gen, 100, 1000);
		vector<VectorXd> samples(n);
		for (int j = 0; j < n; j++) {
			if (i == 2) // Try using the static version
				mvnrnd::sample(samples[j], mu, Sigma, intRand(gen, INT_MIN, INT_MAX));
			else
				generator.sample(samples[j]);
		}
		// Put samples into MatrixXd, so we can give it to matlab easily
		MatrixXd m(n, 2);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < 2; j++)
				m(i, j) = samples[i][j];
		}

		/*
		addVar(m, "xy");
		matlab("subplot(3,1,1)");
		matlab("scatter(xy(:,1),xy(:,2), 5, 'r')");
		matlab("subplot(3,1,2)");
		matlab("scatter(xy(:,1),xy(:,2), 5, 'r')");

		// Now do the same thing on matlab's side. End with three plots - C++, (C++ with Matlab), Matlab.
		addVar(mu, "mu");
		addVar(Sigma, "Sigma");
		addVar(n, "n");
		matlab("xy = mvnrnd(mu, Sigma, n)");
		matlab("hold on");
		matlab("scatter(xy(:,1),xy(:,2), 5, 'b')");
		matlab("subplot(3,1,3)");
		matlab("scatter(xy(:,1),xy(:,2), 5, 'b')");

		cout << "Press enter for next." << endl;
		forceGetchar();
		matlab("close all");
		*/
		cout << "See source - there is matlab code for plotting that must be updated to gnuplot." << endl;
		forceGetchar();
	}

	cout << "test_mvnrnd done. Press enter to continue." << endl;
	forceGetchar();
	//matlab("close all");
}
