#include <Zaffre/headers/Zaffre.hpp>

void plotMatrix(const MatrixXd & m, bool use3D) {
	const char * Zaffre_Path = getenv("ZAFFRE_PATH");
	if (Zaffre_Path == nullptr)
	{
		errorExit("Error getting environment variable ZAFFRE_PATH.");
	}
	if (use3D)
	{
		ofstream out((string)((string)Zaffre_Path + "/temp/plotMatrix3D.temp").c_str());
		out << m.transpose() << endl;   // The way gnuplot is set up right now, we need to transpose for the value function to come out right
		out.close();
		cout << "Press enter to continue." << endl;
		string s = "(cd " + (string)Zaffre_Path + "/gnuplot_code; gnuplot -persist plotMatrix3D)";
		(void)(system(s.c_str())+1);
	}
	else
	{
		ofstream out((string)((string)Zaffre_Path + "/temp/plotMatrix.temp").c_str());
	    out << m.transpose() << endl;   // The way gnuplot is set up right now, we need to transpose for the value function to come out right
		out.close();
		cout << "Press enter to continue." << endl;
		string s = "(cd " + (string)Zaffre_Path + "/gnuplot_code; gnuplot -persist plotMatrix)";
		(void)(system(s.c_str())+1);
	}
}

void plot2Matrices(const MatrixXd & m1, const MatrixXd & m2) {
	const char * Zaffre_Path = getenv("ZAFFRE_PATH");
	if (Zaffre_Path == nullptr)
	{
		errorExit("Error getting environment variable ZAFFRE_PATH.");
	}
	ofstream out1((string)((string)Zaffre_Path + "/temp/plot2Matrices1.temp").c_str());
	ofstream out2((string)((string)Zaffre_Path + "/temp/plot2Matrices2.temp").c_str());
    	out1 << m1.transpose() << endl;   // The way gnuplot is set up right now, we need to transpose for the value function to come out right
    	out2 << m2.transpose() << endl;   // The way gnuplot is set up right now, we need to transpose for the value function to come out right
	out1.close();
	out2.close();
	cout << "Press enter to continue." << endl;
	string s = "(cd " + (string)Zaffre_Path + "/gnuplot_code; gnuplot -persist plot2Matrices)";
	(void)(system(s.c_str())+1);
}

LinePlot createPlot(const string & title,
					const string & xLabel,
					const string & yLabel,
					const VectorXi & xValues,
					const string & spaceSeparatedNames,
					int numLines...)
{
	LinePlot l;
	l.setTitle(title);
	l.setXLabel(xLabel);
	l.setYLabel(yLabel);
	VectorXd xValues2(xValues.size());						// Plot requires a VectorXd
	for (int i = 0; i < (int)xValues.size(); i++)
		xValues2[i] = xValues[i];
	l.setXValues(xValues2);

	// Split names into vector
	stringstream ss(spaceSeparatedNames);
	istream_iterator<string> begin(ss);
	istream_iterator<string> end;
	vector<string> names(begin, end);
	l.setLegend(names);

	// Get data
	vector<VectorXd> data(numLines);
	va_list args;
	va_start(args, numLines);
	for (int i = 0; i < numLines; i++) {
		 vector<VectorXd> *cur = va_arg(args, vector<VectorXd>*);
		 VectorXd curMean((*cur)[0].size());
		 for (int j = 0; j < (int)curMean.size(); j++) {
			curMean[j] = 0;
			for (int k = 0; k < (int)cur->size(); k++)
				curMean[j] += (*cur)[k][j];
			curMean[j] /= (double)cur->size();
		 }
		 data[i] = curMean;
	}
	l.setData(data);
	return l;
}


LinePlot::LinePlot() {
	m.resize(0);
	title="";
	xLabel="";
	yLabel="";
	names.resize(0);
	errorBars.resize(0);
	logX = false;
	logY = false;
}

void LinePlot::setData(const vector<VectorXd> & m) {
	this->m.resize(m.size());
	for (int i = 0; i < (int)m.size(); i++)
		this->m[i] = m[i];
}

void LinePlot::setData(const VectorXd & v) {
	m.resize(1);
	m[0] = v;
}

void LinePlot::setXValues(const VectorXd & xValues) {
	this->xValues.resize(1);
	this->xValues[0] = xValues;
}

void LinePlot::setXValues(const vector<VectorXd> & xValues) {
	this->xValues.resize(xValues.size());
	for (int i = 0; i < (int)xValues.size(); i++) {
		this->xValues[i] = xValues[i];
	}
}

void LinePlot::setTitle(const string & title) {
	this->title = title;
}

void LinePlot::setXLabel(const string & xLabel) {
	this->xLabel = xLabel;
}

void LinePlot::setYLabel(const string & yLabel) {
	this->yLabel = yLabel;
}

void LinePlot::setLegend(const string & name) {
	names.resize(1);
	names[0] = name;
}

void LinePlot::setLegend(const vector<string> & names) {
	this->names.resize(names.size());
	for (int i = 0; i < (int)names.size(); i++)
		this->names[i] = names[i];
}

void LinePlot::setErrorBars(const vector<VectorXd> & errorBars) {
	this->errorBars.resize(errorBars.size());
	for (int i = 0; i < (int)errorBars.size(); i++)
		this->errorBars[i] = errorBars[i];
}

void LinePlot::setErrorBars(const VectorXd & errorBars) {
	this->errorBars.resize(1);
	this->errorBars[0] = errorBars;
}

void LinePlot::setLogX() {
	logX = true;
}

void LinePlot::setLogY() {
	logY = true;
}

void LinePlot::draw() {
	// The number of lines is controlled by m
	int numLines = (int)m.size();
	// Make sure that xValues is the right size
	if ((int)xValues.size() == numLines) {
		for (int i = 0; i < numLines; i++) {
			if (xValues[i].size() != m[i].size()) {
				cerr << "Error in LinePlot::draw - xValues not the right length" << endl;
				return;
			}
		}
	}
	else if ((int)xValues.size() == 1) {
		// Make sure all the right size
		for (int i = 0; i < numLines; i++) {
			if ((int)m[i].size() != (int)xValues[0].size()) {
				cerr << "Error in LinePlot::draw - xValues not the right size or length." << endl;
				return;
			}
		}
		// If we get here, it's the right size. Copy over to all
		while ((int)xValues.size() != numLines) {
			xValues.push_back(xValues[xValues.size()-1]);
		}
	}
	else if ((int)xValues.size() == 0) {
		// Default values.
		xValues.resize(m.size());
		for (int i = 0; i < numLines; i++) {
			xValues[i].resize(m[i].size());
			for (int j = 0; j < (int)m[i].size(); j++) {
				xValues[i][j] = j+1;
			}
		}
	}
	else {
		cout << "Error in LinePlot::draw - xValues not set correctly." << endl;
		return;
	}

	// Make sure names is the right size
	while ((int)names.size() < numLines)
		names.push_back(((string)"Line_"+to_string(names.size()+1)).c_str());
	if ((int)names.size() > numLines)
		names.resize(numLines);

	// Make sure that errorBars is the right size
	if ((int)errorBars.size() == 1) {
		// Make sure all the right size
		for (int i = 0; i < numLines; i++) {
			if ((int)m[i].size() != (int)errorBars[0].size()) {
				cerr << "Error in LinePlot::draw - errorBars not the right size or length." << endl;
				return;
			}
		}
		// If we get here, it's the right size. Copy over to all
		while ((int)errorBars.size() != numLines) {
			errorBars.push_back(errorBars[errorBars.size()-1]);
		}
	}
	else if (((int)errorBars.size() != 0) && ((int)errorBars.size() != numLines)) {
		cerr << "Error in LinePlot::draw - errorBars are not the right length." << endl;
		return;
	}
	else if ((int)errorBars.size() != 0) {
		for (int i = 0; i < numLines; i++) {
			if (errorBars[i].size() != m[i].size()) {
				cerr << "Error in LinePlot::draw - errorBars not the right length." << endl;
				return;
			}
		}
	}

	// If logX, transform the x-values
	vector<vector<bool>> plotPoint(numLines);
	vector<bool> plotLine1(numLines), plotLine2(numLines);
	for (int i = 0; i < numLines; i++) {
		plotLine1[i] = true;
		plotLine2[i] = true;
		plotPoint[i].resize(m[i].size());
		for (int j = 0; j < (int)m[i].size(); j++)
			plotPoint[i][j] = true;
	}
	if (logX) {
		for (int i = 0; i < numLines; i++) {
			plotLine1[i] = false;
			for (int j = 0; j < (int)xValues[i].size(); j++) {
				if (xValues[i][j] <= 0) {
					plotPoint[i][j] = false;
				}
				else {
					xValues[i][j] = log10(xValues[i][j]);
					plotLine1[i] = true;
				}
			}
		}
	}
	if (logY) {
		for (int i = 0; i < numLines; i++) {
			plotLine2[i] = false;
			for (int j = 0; j < (int)m[i].size(); j++) {
				if (m[i][j] <= 0)
					plotPoint[i][j] = false;
				else {
					m[i][j] = log10(m[i][j]);
					plotLine2[i] = true;
				}
			}
		}
	}

	int numLinesToCut = 0;
	for (int i = 0; i < numLines; i++) {
		if ((plotLine1[i] == false) || (plotLine2[i] == false))
			numLinesToCut++;
	}
	if (numLinesToCut == numLines) {
		cout << "No lines to plot (are all points cut due to log axes?)." << endl;
		return;
	}

	// Plot!
	const char * Zaffre_Path = getenv("ZAFFRE_PATH");
	if (Zaffre_Path == nullptr) {
		errorExit("Error getting environment variable ZAFFRE_PATH.");
	}
	ofstream out((string)((string)Zaffre_Path + "/temp/linePlot.temp").c_str());
	for (int i = 0; i < (int)m.size(); i++) {
		if ((plotLine1[i] == false) || (plotLine2[i] == false))
			continue; // Don't plot this line.
		out << "\"" << names[i] << "\"" << endl;
		for (int j = 0; j < (int)m[i].size(); j++) {
			if (plotPoint[i][j])
				out << xValues[i][j] << "\t" << m[i][j] << endl;
		}
		out << endl << endl;
	}
	cout << "Press enter to continue." << endl;
	string s = "(cd " + (string)Zaffre_Path + "/gnuplot_code; gnuplot -persist -e \"datafile='../temp/linePlot.temp'; title_string='" + title + "'; xlabel_string='" + xLabel + "'; ylabel_string='" + yLabel + "'; numLines='" + to_string(numLines-numLinesToCut) + "'\" linePlot)";
	(void)(system(s.c_str())+1);
}
