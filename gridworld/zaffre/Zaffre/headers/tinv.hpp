#ifndef _TINV_HPP_
#define _TINV_HPP_

#include <Zaffre/headers/Zaffre.hpp>

// Loads the globals.
void init_tinv();

// Only thread-safe if you already called init_tinv()
// See Matlab's documentation for tinv, where we use "df" rather than "nu" to denote the number of degrees of freedom
double tinv(const double & p, const double & df);

void testTinv();

#endif
