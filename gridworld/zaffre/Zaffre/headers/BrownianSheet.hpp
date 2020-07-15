#ifndef _BROWNIANSHEET_HPP_
#define _BROWNIANSHEET_HPP_

#include <Zaffre/headers/Zaffre.hpp>

//////////////////////////////////////////////////
///// Function prototypes
//////////////////////////////////////////////////

// Roughness parameter alpha = 2H. Use H=0.5 to get a standard brownian sheet.
MatrixXd BrownianSheet(const int & rows, const int & cols, const double H, const unsigned long & RNGSeed);

// Get brownian sheet as a vector
VectorXd BrownianSheet_Vec(const int & rows, const int & cols, const double H, const unsigned long & RNGSeed);

#endif // _BROWNIANSHEET_HPP_
