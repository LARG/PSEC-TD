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

/*
Function for solving a QP on the simplex using Gurobi
Minimize x'Ax
Subject to x >= 0
And sum(x) = 1

Assumes that A is square.
*/
VectorXd solveQP_onSimplex(const MatrixXd & A, GRBEnv * env)
{
	int n = (int)A.rows();
	bool createdEnv = false; // Track whether or not we created it so that we know if we should delete it
	VectorXd result(n);

	try
	{
		// If no GRBEnv provided, create one
		if (env == nullptr)
		{
			createdEnv = true;
			env = new GRBEnv();						// If a Gurobi environment hasn't been provided, create one. There should usually only be one of these in the program
			env->set(GRB_IntParam_LogToConsole, 0); // Don't print all the debug info
		}
		GRBModel model = GRBModel(*env);			// Create this optimization problem

		// Create the variables
		vector<GRBVar> vars(n);
		for (int i = 0; i < n; i++)
			vars[i] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, ((string)("x_") + to_string(i)).c_str());

		model.update();								// Integrate new variables

		// Create the objective expression
		GRBQuadExpr obj = 0.0;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				obj += vars[i] * vars[j] * A(i, j);
		}

		// Set it as the objective
		model.setObjective(obj, GRB_MINIMIZE);

		// Add constraints. We already have that all variables are in [0,1]. Now say that that sum to one
		GRBLinExpr sum = 0;
		for (int i = 0; i < n; i++)
			sum += vars[i];
		model.addConstr(sum, GRB_EQUAL, 1, "simplexConstraint");

		// Optimize model
		model.optimize();

		for (int i = 0; i < n; i++)
			result[i] = vars[i].get(GRB_DoubleAttr_X);
	}
	catch (GRBException e)
	{
		cout << "Gurobi Error code = " << e.getErrorCode() << endl;
		cout << e.getMessage() << endl;
		getchar();
		getchar();
		exit(1);
	}
	catch (...)
	{
		cout << "Gurobi Exception during optimization." << endl;
		getchar();
		getchar();
		exit(1);
	}

	// Clean up memory
	if (createdEnv)
		delete env;		// This should happen after the "try" statement, otherwise Gurobi throws a warning about the environment being deleted too soon.

	return result;
}
