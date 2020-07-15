#ifndef _PLOT_HPP_
#define _PLOT_HPP_

#include <Zaffre/headers/Zaffre.hpp>

void plotMatrix(const MatrixXd & m, bool use3D = false);

void plot2Matrices(const MatrixXd & m1, const MatrixXd & m2);

// Class for plotting several lines
class LinePlot {
public:
	LinePlot();
	void setData(const vector<VectorXd> & m);	// For plotting several lines
	void setData(const VectorXd & v);			// For plotting one line
	void setXValues(const VectorXd & xValues);
	void setXValues(const vector<VectorXd> & xValues);
	void setTitle(const string & title);
	void setXLabel(const string & xLabel);
	void setYLabel(const string & yLabel);
	void setLegend(const string & name);		// For a single line, or if the same for all lines
	void setLegend(const vector<string> & names);
	void setErrorBars(const vector<VectorXd> & errorBars);	// For several lines
	void setErrorBars(const VectorXd & errorBars);			// For a single line (or if the same for all lines)
	void setLogX();								// Can lose data by doing this (throws away points <= 0) so it's a one-way operation
	void setLogY();								// Can lose data by doing this (throws away points <= 0) so it's a one-way operation
	void draw();								// Actually create the plot
private:
	vector<VectorXd> m;
	vector<VectorXd> xValues;
	string title;
	string xLabel;
	string yLabel;
	vector<string> names;
	vector<VectorXd> errorBars;
	bool logX;
	bool logY;
};

// Variadic plot function for Phil's most common use case
LinePlot createPlot(const string & title,
					const string & xLabel,
					const string & yLabel,
					const VectorXi & ,
					const string & spaceSeparatedNames,
					int numLines...); // After numlines comes the data, each line as a matrix

//void asynchDraw(LinePlot & L);

#endif // _PLOT_HPP_
