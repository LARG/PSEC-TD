#include <Zaffre/headers/Zaffre.hpp>

mvnpdf::mvnpdf(const VectorXd & meanVec, const MatrixXd & covarMat) {
	n = (int)meanVec.size();
	setMean(meanVec);
	setCovar(covarMat);
}

void mvnpdf::setCovar(const MatrixXd & covarMat) {
	assert((int)covarMat.rows() == n); // If not, make a new object
	assert((int)covarMat.cols() == n); // If not, make a new object
	Sigma = covarMat;
	SigmaInv = pinv(Sigma);
	DetermSigma = Sigma.determinant();
}

void mvnpdf::setMean(const VectorXd & meanVec) {
	assert((int)meanVec.size() == n); // If not, make a new object
	mu = meanVec;
}

// See http://www.cs.huji.ac.il/~csip/tirgul34.pdf
double mvnpdf::Pr(const VectorXd & x) {
	double firstTerm = 1.0 / (pow(2.0*M_PI, (double)n / 2.0)*sqrt(DetermSigma));
	VectorXd temp = x - mu;
	double numerator = temp.transpose()*SigmaInv*temp;
	double exponent = -numerator / 2.0;
	return firstTerm * exp(exponent);
}

double mvnpdf::Pr(const VectorXd & x, VectorXd & meanVec, const MatrixXd & covarMat) {
	int n = (int)meanVec.size();
	VectorXd mu;
	MatrixXd Sigma;
	MatrixXd SigmaInv;
	double DetermSigma;

	// setMean(meanVec);
	assert((int)meanVec.size() == n); // If not, make a new object
	mu = meanVec;

	// setCovar(covarMat);
	assert((int)covarMat.rows() == n); // If not, make a new object
	assert((int)covarMat.cols() == n); // If not, make a new object
	Sigma = covarMat;
	SigmaInv = pinv(Sigma);
	DetermSigma = Sigma.determinant();

	// Actually compute Pr
	double firstTerm = 1.0 / (pow(2.0*M_PI, (double)n / 2.0)*sqrt(DetermSigma));
	VectorXd temp = x - mu;
	double numerator = temp.transpose()*SigmaInv*temp;
	double exponent = -numerator / 2.0;
	return firstTerm * exp(exponent);
}

void test_mvnpdf() {
	cout << "Starting test of mvnpdf. C++ and matlab values should be similar." << endl;

	for (int i = 0; i < 3; i++) {
		mt19937_64 gen((int)time(NULL));
		int d = intRand(gen, 1, 4);
		VectorXd mu(d);
		MatrixXd Sigma(d, d);
		for (int i = 0; i < d; i++)
			mu[i] = rand(gen, -10.0, 10.0);

		// Lower Triangular
		for (int r = 0; r < d; r++) {
			for (int c = 0; c < d; c++) {
				if (c > r)
					Sigma(r, c) = 0;
				else if (c == r)
					Sigma(r, c) = rand(gen, 0.1, 10); // Eigenvalues are on the diagonal
				else
					Sigma(r, c) = rand(gen, -10, 10);
			}
		}
		Sigma = Sigma*Sigma.transpose();	// Make it a PSD matrix
		//Sigma = Sigma + 0.00001 * MatrixXd::Identity(d, d); // Make it a PD matrix

		mvnrnd generator(mu, Sigma, (int)time(NULL));
		mvnpdf pdf(mu, Sigma);
		VectorXd sample;

		// Tell matlab about these two
		//addVar(mu, "mu");
		//addVar(Sigma, "Sigma");
		cout << "**** See source for mvnpdf test - needs Matlab." << endl;

		cout << "C++\tMatlab" << endl;
		for (int j = 0; j < 10; j++) {
			generator.sample(sample);
			//double probability = pdf.Pr(sample);
			//if (j % 2 == 0)
				//probability = mvnpdf::Pr(sample, mu, Sigma);
			// Now get the probability from matlab
			//addVar(sample, "sample");
			//matlab("matlabResult = mvnpdf(sample, mu, Sigma);");
			//double matlabProbability = getVar<double>("matlabResult");
			//cout << probability << '\t' << matlabProbability << endl;
		}
		cout << endl << endl;

		cout << "Press enter for next." << endl;
		forceGetchar();
	}

	cout << "test_mvnpdf done. Press enter to continue." << endl;
	forceGetchar();
}
