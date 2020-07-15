#include <Zaffre/headers/Zaffre.hpp>

/*
This is the only file that uses CImg. If you don't have X11, then you could
comment out this file. Maybe later make a compile flag for this, gnuplot, and gurobi
so that you can set which ones you have, and it errors if you try to call them on a machine
without them.
*/

#include <Zaffre/headers/CImg.h>

using namespace cimg_library;                   // For CImg.h (displays images)

/*
Plot using CImg.
    k = number of points to average together for each point in the plot
*/
void plot_CImg(const VectorXd & Z, const char * title, int k)
{
    VectorXd Y(Z.size()/k);
    for (int i = 0; i < (int)Y.size(); i++)
        Y[i] = Z.block(i*k,0,k,1).mean();
    int n = (int)Y.size(), width = 1000, height = 500;
    double xScale = width/n, yScale = height / (Y.maxCoeff() - Y.minCoeff());
    CImg<unsigned char> image(width, height, 1, 3, 1);
    const unsigned char color[] = { 255,255,255 };
    for (int i = 0; i < n-1; i++) {
        int x0 = (int)(i*xScale), x1 = (int)((i+1)*xScale),
            y0 = (int)((Y[i]-Y.minCoeff())*yScale), y1 = (int)((Y[i+1]-Y.minCoeff())*yScale);
        image.draw_line(x0,height-y0,x1,height-y1, color);
    }
    string s = (string)"(1," + (string)xtoa(Y.maxCoeff()) + (string)")";
    image.draw_text(0, 0, s.c_str(), color);
    s = (string)"(1," + (string)xtoa(Y.minCoeff()) + (string)")";
    image.draw_text(0, height-15, s.c_str(), color);
    s = (string)"(" + (string)(string)xtoa(n) + (string)"," + (string)xtoa(Y.minCoeff()) + (string)")";
    image.draw_text(width-75, height-15, s.c_str(), color);
    s = (string)"(" + (string)(string)xtoa(n) + (string)"," + (string)xtoa(Y.maxCoeff()) + (string)")";
    image.draw_text(width-75, 0, s.c_str(), color);

    CImgDisplay display(image, title);
    cout << "Press enter to close display window." << endl;
    getchar();
}
