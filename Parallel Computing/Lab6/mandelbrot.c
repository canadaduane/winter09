#include <stdio.h>

double x_max = 500;
double y_max = 500;
int i_max = 1000;

int mb(double x_not, double y_not, double mag)
{
    double cx = (x_not / x_max - 0.5) / mag * 3.0 - 0.75;
    double cy = (y_not / y_max - 0.5) / mag * 3.0;
    double x = 0.0, y = 0.0;
    double x_sq, y_sq;
    int i = 0;
    while ((x_sq = x*x) + (y_sq = y*y) <= 100.0 && i++ < i_max)
    {
        double xtemp = x_sq - y_sq + cx;
        y = 2 * x * y + cy;
        x = xtemp;
    }
    if (i >= i_max) return 0;
    else           return 1;
}

// int main(int argc, char** argv)
// {
//     int x, y;
//     for (y = 0; y < 500; y++)
//     {
//         for (x = 0; x < 500; x++)
//         {
//             printf("%d ", mb((double)x, (double)y, 2.0));
//         }
//         printf("\n");
//     }
// }
