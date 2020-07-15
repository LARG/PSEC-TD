#include <Zaffre/headers/Zaffre.hpp>

double ** Zaffre_TTable_Global = nullptr;
double * Zaffre_TTable_P_Global = nullptr;
double * Zaffre_TTable_DF_Global = nullptr;
int * Zaffre_TTable_NumP_Global = nullptr;
int * Zaffre_TTable_NumDF_Global = nullptr;

// Loads the globals.
void init_tinv() {
	if (Zaffre_TTable_NumP_Global != nullptr)
		return; // Already initialized

	const char * Zaffre_Path = getenv("ZAFFRE_PATH");
	if (Zaffre_Path == nullptr)
	{
		errorExit("Error getting environment variable ZAFFRE_PATH.");
	}
	ifstream	inTable((string)((string)Zaffre_Path + "/TTable/ttable.csv").c_str()),
			inP((string)((string)Zaffre_Path + "/TTable/ttable_p.csv").c_str()),
			inDF((string)((string)Zaffre_Path + "/TTable/ttable_df.csv").c_str());
	if (!inTable.good())
		errorExit("Failed to open ttable.csv.\nDid you set the ZAFFRE_PATH Environment variable?");
	if (!inP.good())
		errorExit("Failed to open ttable_p.csv\nDid you set the ZAFFRE_PATH Environment variable?");
	if (!inDF.good())
		errorExit("Failed to open ttable_df.csv\nDid you set the ZAFFRE_PATH Environment variable?");
	vector<string> p_asStrings = csv_getNextLineAndSplitIntoTokens(inP),
		df_asStrings = csv_getNextLineAndSplitIntoTokens(inDF);

	Zaffre_TTable_NumP_Global = new int;
	*Zaffre_TTable_NumP_Global = (int)p_asStrings.size();
	Zaffre_TTable_NumDF_Global = new int;
	*Zaffre_TTable_NumDF_Global = (int)df_asStrings.size();

	// Now store in more handy names:
	int numP = (int)p_asStrings.size(), numDF = (int)df_asStrings.size();

	// Load the tables
	Zaffre_TTable_P_Global = new double[numP];
	for (int i = 0; i < numP; i++)
		Zaffre_TTable_P_Global[i] = atof(p_asStrings[i].c_str());

	Zaffre_TTable_DF_Global = new double[numDF];
	for (int i = 0; i < numDF; i++)
		Zaffre_TTable_DF_Global[i] = atof(df_asStrings[i].c_str());

	Zaffre_TTable_Global = new double*[numDF];
	vector<string> row;
	for (int i = 0; i < numDF; i++) {
		Zaffre_TTable_Global[i] = new double[numP];
		row = csv_getNextLineAndSplitIntoTokens(inTable);
		for (int j = 0; j < numP; j++)
			Zaffre_TTable_Global[i][j] = atof(row[j].c_str());
	}
}

// Only thread-safe if you already called init_tinv()
// See Matlab's documentation for tinv, where we use "df" rather than "nu" to denote the number of degrees of freedom
double tinv(const double & p, const double & df) {
	init_tinv();	// Make sure the tables are loaded

	// Check if we have the p-value
	int i, numP = *Zaffre_TTable_NumP_Global, numDF = *Zaffre_TTable_NumDF_Global;
	for (i = 0; i < numP; i++) {
		if (p == Zaffre_TTable_P_Global[i])
			break;
	}
	if (i != numP) {
		// We have the p-value exactly. Check if we have the df-value exactly
		int j;
		for (j = 0; j < numDF; j++) {
			if (df == Zaffre_TTable_DF_Global[j])
				break;
		}
		if (j != numDF)
			return Zaffre_TTable_Global[j][i]; // We have both exactly!
		// If we get here, we know the p-value, but we don't know the df
		int j1 = 0, j2;
		for (j = 1; j < numDF; j++) {
			if (fabs(Zaffre_TTable_DF_Global[j] - df) < fabs(Zaffre_TTable_DF_Global[j1] - df))
				j1 = j;
		}
		// j1 is now the closest index. Load j2 with the next closest
		j2 = (j1 == 0 ? 1 : 0);
		for (j = 0; j < numDF; j++) {
			if (j == j1) continue;
			if (fabs(Zaffre_TTable_DF_Global[j] - df) < fabs(Zaffre_TTable_DF_Global[j2] - df))
				j2 = j;
		}
		// j1 and j2 are now the two closest indices. Do a linear interpretation between their values.
		if (j2 < j1)
			swap(j1, j2); // Make sure j1 is the lesser of the two.
		double dfJ1 = Zaffre_TTable_DF_Global[j1], dfJ2 = Zaffre_TTable_DF_Global[j2]; // Load dfJ1 and dfJ2 with the # degrees of freedom from those two entries
		double dfJ1Result = Zaffre_TTable_Global[j1][i], dfJ2Result = Zaffre_TTable_Global[j2][i];
		double slope = (dfJ2Result - dfJ1Result) / (dfJ2 - dfJ1);
		return (df - dfJ1)*slope + dfJ1Result;
	}

	// If we get here, we don't have the p-value --- check if we have the df-value
	for (i = 0; i < numDF; i++) {
		if (df == Zaffre_TTable_DF_Global[i])
			break;
	}
	if (i != numDF) { // We have the df-value but not the p-value
		int j1 = 0, j2, j;
		for (j = 1; j < numP; j++) {
			if (fabs(Zaffre_TTable_P_Global[j] - p) < fabs(Zaffre_TTable_P_Global[j1] - p))
				j1 = j;
		}
		// j1 is now the closest index. Load j2 with the next closest
		j2 = (j1 == 0 ? 1 : 0);
		for (j = 0; j < numP; j++) {
			if (j == j1) continue;
			if (fabs(Zaffre_TTable_P_Global[j] - p) < fabs(Zaffre_TTable_P_Global[j2] - p))
				j2 = j;
		}
		// j1 and j2 are now the two closest indices. Do a linear interpretation between their values.
		if (j2 < j1)
			swap(j1, j2); // Make sure j1 is the lesser of the two.
		double pJ1 = Zaffre_TTable_P_Global[j1], pJ2 = Zaffre_TTable_P_Global[j2];
		double pJ1Result = Zaffre_TTable_Global[i][j1], pJ2Result = Zaffre_TTable_Global[i][j2];
		double slope = (pJ2Result - pJ1Result) / (pJ2 - pJ1);
		return (p - pJ1)*slope + pJ1Result;
	}

	// If we get here, we don't have the p-value or the df-value. Find the four nearest points and do a planar fit.
	int i1 = 0, i2;
	for (i = 1; i < numDF; i++) {
		if (fabs(Zaffre_TTable_DF_Global[i] - df) < fabs(Zaffre_TTable_DF_Global[i1] - df))
			i1 = i;
	}
	i2 = (i1 == 0 ? 1 : 0);
	for (i = 0; i < numDF; i++) {
		if (i == i1) continue;
		if (fabs(Zaffre_TTable_DF_Global[i] - df) < fabs(Zaffre_TTable_DF_Global[i2] - df))
			i2 = i;
	}
	if (i2 < i1)
		swap(i1, i2);

	int j1 = 0, j2, j;
	for (j = 1; j < numP; j++) {
		if (fabs(Zaffre_TTable_P_Global[j] - p) < fabs(Zaffre_TTable_P_Global[j1] - p))
			j1 = j;
	}
	// j1 is now the closest index. Load j2 with the next closest
	j2 = (j1 == 0 ? 1 : 0);
	for (j = 0; j < numP; j++) {
		if (j == j1) continue;
		if (fabs(Zaffre_TTable_P_Global[j] - p) < fabs(Zaffre_TTable_P_Global[j2] - p))
			j2 = j;
	}
	// j1 and j2 are now the two closest indices. Do a linear interpretation between their values.
	if (j2 < j1)
		swap(j1, j2); // Make sure j1 is the lesser of the two.

	// i1 and i2 are the two closest indices for df
	// j1 and j2 are the two closest indices for p
	MatrixXd A(4, 3);
	VectorXd b(4);
	A(0, 0) = Zaffre_TTable_P_Global[j1];
	A(0, 1) = Zaffre_TTable_DF_Global[i1];
	A(0, 2) = 1;
	b(0) = Zaffre_TTable_Global[i1][j1];

	A(1, 0) = Zaffre_TTable_P_Global[j2];
	A(1, 1) = Zaffre_TTable_DF_Global[i1];
	A(1, 2) = 1;
	b(1) = Zaffre_TTable_Global[i1][j2];

	A(2, 0) = Zaffre_TTable_P_Global[j1];
	A(2, 1) = Zaffre_TTable_DF_Global[i2];
	A(2, 2) = 1;
	b(2) = Zaffre_TTable_Global[i2][j1];

	A(3, 0) = Zaffre_TTable_P_Global[j2];
	A(3, 1) = Zaffre_TTable_DF_Global[i2];
	A(3, 2) = 1;
	b(3) = Zaffre_TTable_Global[i2][j2];

	// Solve for coefficients x for the plane that fits through the four points
	VectorXd x = linsolve(A, b);
	return p*x[0] + df*x[1] + x[2];
}

void testTinv() {
	cout << "tinv(0.95,5) = " << tinv(0.95, 5) << endl;
	cout << "tinv(0.95,102) [df missing] = " << tinv(0.95, 102) << endl;
	cout << "tinv(0.81,10) [p missing] = " << tinv(0.81, 10) << endl;
	cout << "tinv(0.81,102) [df and p missing] = " << tinv(0.81, 102) << endl;
	cout << "Done. Press enter to continue." << endl;
	forceGetchar();
}
