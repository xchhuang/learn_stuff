// References:
// https://github.com/pybind/python_example


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define M_PI 3.14159265358979323846


std::vector<float> precompute_boundary(int N, int width, int height, int nbbins, int n_rmax) {
    // int nbbins = 20;
    // int n_rmax = 2;
    float rmax = 2.0 * sqrt(1.0 / (2 * sqrt(3) * N));
    // int width = 256;
    // int height = 256;
    float bnd_x = 1.0 / (width + 1);
    float bnd_y = 1.0 / (height + 1);
    float step_x = bnd_x;
    float step_y = bnd_y;

    std::vector<float> grid_pts;

    for (int i = 0; i < width; i ++) {
        float x_pos = bnd_x + step_x * i;
        for (int j = 0; j < height; j++) {
            float y_pos = bnd_y + step_y * j;
            int index = i * height + j;
            // std::cout << "xy: " << x_pos << " " << y_pos << std::endl;
            grid_pts.push_back(x_pos);
            grid_pts.push_back(y_pos);
        }
    }

    // std::cout << grid_pts.size() << std::endl;

    std::vector<float> boundary_term;
    for (int j = 0; j < width * height * nbbins; j++) {
        boundary_term.push_back(0.0);
    }

    float stepsize = (float)n_rmax / nbbins;
    
    // std::cout << "stepsize: " << stepsize << " " << rmax << std::endl; 
    std::vector<float> radii;
    for (int i = 1; i < nbbins + 1; i++) {
        radii.push_back(i * stepsize * rmax);
        // std::cout << "radii[i]: " << radii[i-1] << std::endl;
    }

    std::vector<float> full_angle(nbbins, 1.0);
    std::vector<float> weights(nbbins, 0.0);

    for (int i=0; i<width * height; i++) {
        
        for (int j=0; j<nbbins; j++) {
            full_angle[j] = 2 * M_PI;
        }
        float x = grid_pts[i*2+0];
        float y = grid_pts[i*2+1];

        float dx = x;
        float dy = y;
        for (int j = 0; j < nbbins; j++) {
            if (radii[j] > dx) {
                float alpha = acos(dx / radii[j]);
                full_angle[j] -= std::min(alpha, atan2(dy, dx)) + std::min(alpha, atan2(1 - dy, dx));
            }
        }

        dx = 1 - x;
        dy = y;
        for (int j = 0; j < nbbins; j++) {
            if (radii[j] > dx) {
                float alpha = acos(dx / radii[j]);
                full_angle[j] -= std::min(alpha, atan2(dy, dx)) + std::min(alpha, atan2(1 - dy, dx));
            }
        }

        dx = y;
        dy = x;
        for (int j = 0; j < nbbins; j++) {
            if (radii[j] > dx) {
                float alpha = acos(dx / radii[j]);
                full_angle[j] -= std::min(alpha, atan2(dy, dx)) + std::min(alpha, atan2(1 - dy, dx));
            }
        }

        dx = 1 - y;
        dy = x;
        for (int j = 0; j < nbbins; j++) {
            if (radii[j] > dx) {
                float alpha = acos(dx / radii[j]);
                full_angle[j] -= std::min(alpha, atan2(dy, dx)) + std::min(alpha, atan2(1 - dy, dx));
            }
        }

        for (int j = 0; j < nbbins; j++) {
            float c = full_angle[j] / (2 * M_PI);
            full_angle[j] = std::fmax(0, std::fmin(c, 1));    // clamp(x, 0, 1)
            if (full_angle[j] > 0) {
                full_angle[j] = 1.0 / full_angle[j];
            }
            boundary_term[i * nbbins + j] = full_angle[j];
        }
        
    }
    return boundary_term;
}

namespace py = pybind11;

PYBIND11_MODULE(python_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: python_example

        .. autosummary::
           :toctree: _generate

           precompute_boundary
    )pbdoc";

    m.def("precompute_boundary", &precompute_boundary, R"pbdoc(
        Precompute the boundary term for PCF computation
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
