#ifndef _GUROBI_HPP_
#define _GUROBI_HPP_

/*
To get Gurobi to compile, when installed to C:\gurobi605 or /opt/gurobi605/linux64/

Code::Blocks on Ubuntu 14:
	Project
		Build-options
			linker settings
				Link libraries (copy to all settings - debug, release, main project)
					/opt/gurobi605/linux64/lib/libgurobi_c++.a
					/opt/gurobi605/linux64/lib/libgurobi60.so
			search directories
				compiler (for all settings - debug, release, main project)
					/opt/gurobi605/linux64/include
*/

#include <Zaffre/headers/Zaffre.hpp>

#include "gurobi_c++.h"

/*
Function for solving a QP on the simplex using Gurobi
Minimize x'Ax
Subject to x >= 0
And sum(x) = 1

Assumes that A is square.
*/
VectorXd solveQP_onSimplex(const MatrixXd & A, GRBEnv * env = nullptr);

#endif // _GUROBI_HPP_
