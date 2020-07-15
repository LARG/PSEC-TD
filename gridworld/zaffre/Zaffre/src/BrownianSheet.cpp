#include <Zaffre/headers/Zaffre.hpp>

// The corners are filled. Fill the rest. Distribution is a standard normal distribution
void BrownianSheetHelper(MatrixXd & m, mt19937_64 & generator, normal_distribution<double> & distribution,
	bool leftDone, bool rightDone, bool topDone, bool bottomDone, const double & H) {
	int rows = (int)m.rows(), cols = (int)m.cols();

	if (max(rows, cols) <= 2)
		return; // The corners are already filled - we're done!
	if (min(rows, cols) == 2) {
		// We are almost done. It is 2x3 or 3x2
		if (rows == 3) {
			if (!leftDone) {
				m(1, 0) = (m(0, 0) + m(2, 0)) / 2.0 + distribution(generator);
			}
			if (!rightDone) {
				m(1, 1) = (m(0, 1) + m(2, 1)) / 2.0 + distribution(generator);
			}
			return;
		}
		else {
			if (!topDone) {
				m(0, 1) = (m(0, 0) + m(0, 2)) / 2.0 + distribution(generator);
			}
			if (!bottomDone) {
				m(1, 1) = (m(1, 0) + m(1, 2)) / 2.0 + distribution(generator);
			}
			return;
		}
	}
	// If we get here, we can do a full recursion
	// Fill in the column midpoints
	int mc = cols / 2;
	double sigmaC = pow((cols - 1)/2.0, H);
	if (!topDone)
		m(0, mc) = (m(0, 0) + m(0, cols - 1)) / 2.0 + sigmaC*distribution(generator);
	if (!bottomDone)
		m(rows - 1, mc) = (m(rows - 1, 0) + m(rows - 1, cols - 1)) / 2.0 + sigmaC*distribution(generator);

	// Fill in the row midpoints
	int mr = rows / 2;
	double sigmaR = pow((rows - 1) / 2.0, H);
	if (!leftDone)
		m(mr, 0) = (m(0, 0) + m(rows - 1, 0)) / 2.0 + sigmaR*distribution(generator);
	if (!rightDone)
		m(mr, cols - 1) = (m(0, cols - 1) + m(rows-1, cols - 1)) / 2.0 + sigmaR*distribution(generator);

	// Fill in the "center" point
	double sigma = (sigmaR + sigmaC) / 2.0; // Just average the two - they could be slightly different
	m(mr, mc) = (m(0, mc) + m(rows - 1, mc) + m(mr, 0) + m(mr, cols - 1)) / 4.0 + sigma*distribution(generator);

	// Recurse
	MatrixXd temp = m.block(0, 0, mr + 1, mc + 1);
	BrownianSheetHelper(temp, generator, distribution, leftDone, false, topDone, false, H);
	m.block(0, 0, mr + 1, mc + 1) = temp;
	temp = m.block(mr, 0, rows - mr, mc + 1);
	BrownianSheetHelper(temp, generator, distribution, leftDone, false, true, bottomDone, H);
	m.block(mr, 0, rows - mr, mc + 1) = temp;
	temp = m.block(0, mc, mr + 1, cols - mc);
	BrownianSheetHelper(temp, generator, distribution, true, rightDone, topDone, false, H);
	m.block(0, mc, mr + 1, cols - mc) = temp;
	temp = m.block(mr, mc, rows - mr, cols - mc);
	BrownianSheetHelper(temp, generator, distribution, true, rightDone, true, bottomDone, H);
	m.block(mr, mc, rows - mr, cols - mc) = temp;
}

// Roughness parameter alpha = 2H. Use H=0.5 to get a standard brownian sheet.
MatrixXd BrownianSheet(const int & rows, const int & cols, const double H, const unsigned long & RNGSeed) {
	// Make it square - easier to produce.
	int n = max(rows, cols);
	// Make normal distribution
	mt19937_64 generator(RNGSeed);
	normal_distribution<double> distribution(0, 1);
	// Make result, and fill in corners
	MatrixXd result = MatrixXd::Zero(n, n);
	result(0, 0) = distribution(generator);
	result(0, n - 1) = distribution(generator);
	result(n - 1, 0) = distribution(generator);
	result(n - 1, n - 1) = distribution(generator);
	// Call recursive function
	BrownianSheetHelper(result, generator, distribution, false, false, false, false, H);
	// Only return the requested chunk of the result
	MatrixXd cutResult = result.block(0, 0, rows, cols);
	// Make it mean zero
	cutResult.array() -= cutResult.array().mean();
	double variance = 0;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			variance += cutResult(r, c)*cutResult(r, c); // Just squared because mean = 0.
		}
	}
	variance /= (rows + cols - 1);
	cutResult.array() /= sqrt(variance);
	return cutResult;
}

VectorXd BrownianSheet_Vec(const int & rows, const int & cols, const double H, const unsigned long & RNGSeed) {
    MatrixXd bs = BrownianSheet(rows, cols, H, RNGSeed);
    VectorXd result(bs.rows()*bs.cols());
    for (int i = 0; i < (int)result.size(); i++)
        result[i] = bs(i/bs.cols(), i%bs.cols());
    return result;
}
