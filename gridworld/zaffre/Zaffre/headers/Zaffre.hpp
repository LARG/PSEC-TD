#ifndef _ZAFFRE_H_
#define _ZAFFRE_H_

/*
The Zaffre library includes useful functions and includes.
Ideally, for most simple projects you can include Zaffre, and everything
that you need will be there.

Zaffre assumes that you have the following installed:
	0. Add Environment variable ZAFFRE_PATH = the Zaffre
 directory. E.g., ZAFFRE_PATH="/usr/local/include/Zaffre"
		- To permanently add the environment variable, add it to /etc/environment
	1. Eigen: There should be an "Eigen" directory in the same directory as the "Zaffre" folder.
	2. Gurobi: Instructions for installing Gurobi are in Gurobi.hpp
	3. X11: I installed it with: "sudo apt-get install xorg openbox" followed by "sudo apt-get install libx11-dev". Also need "sudo apt-get install gnuplot-x11" not just normal gnuplot. Not sure if the first command was actually needed.
		- Also under linker settings / other linker options I added:  -lpthread -lX11
	4. To allow for writing tempfiles, change the permissions on the Zaffre folder to allow read/write
	5. Compiler settings -> other options -> -fopenmp  (to use parallel for loops).
		- Also add -fopenmp to linker options.
*/

#include <Zaffre/headers/Includes.hpp>					// Common includes
#include <Zaffre/headers/IOStringUtils.hpp>				// Basic file and string I/O functions
#include <Zaffre/headers/LinearAlgebra.hpp>				// Basic linear algebra functions that Eigen lacks (e.g. tests for positive semidefinite and pinv).
#include <Zaffre/headers/MathUtils.hpp>					// Some useful math functions that aren't in other included libraries
#include <Zaffre/headers/Timer.hpp>
#include <Zaffre/headers/Optimization.hpp>
//#include <Zaffre/headers/ConcentrationInequalities.hpp>
#include <Zaffre/headers/mvnrnd.hpp>
#include <Zaffre/headers/mvnpdf.hpp>
#include <Zaffre/headers/tinv.hpp>
#include <Zaffre/headers/BrownianSheet.hpp>
//#include <Zaffre/headers/Gurobi.hpp>
#include <Zaffre/headers/Graphics.hpp>					// Use CImg.h to draw stuff in a window
#include <Zaffre/headers/Plot.hpp>
#include <Zaffre/headers/MarsagliaGenerator.hpp>

void init(bool use_tinv = true);
void close(bool clearTempFiles = true);

#endif // _ZAFFRE_H_
