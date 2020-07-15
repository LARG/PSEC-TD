#ifndef _GRAPHICS_HPP_
#define _GRAPHICS_HPP_

#include <Zaffre/headers/Zaffre.hpp>

/*
This is the only file that uses CImg. If you don't have X11, then you could
comment out this file. Maybe later make a compile flag for this, gnuplot, and gurobi
so that you can set which ones you have, and it errors if you try to call them on a machine
without them.
*/

/*
Plot using CImg.
    k = number of points to average together for each point in the plot
*/
void plot_CImg(const VectorXd & Z, const char * title, int k = 1);

#endif // _GRAPHICS_HPP
