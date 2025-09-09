#include <iostream>
#include <iomanip>
#include <vector>

// we use std::vector for matrix and vector representation

using std::vector;
template <typename T>
using matrix = std::vector<std::vector<T>>;


// define a hilbert matrix of size n x n

template <typename T>
matrix<T> hilbert(int n) {
    matrix<T> H(n, vector<T>(n));
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            H[i][j] = static_cast<T>(1.0) / static_cast<T>(i + j + 1);
        }
    }
    return H;
}

// Matrix-vector multiplication: y = A * x

template <typename T>
std::vector<T> matvec(const matrix<T>& A, const vector<T>& x) {

    int n = A.size();
    std::vector<T> y(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < x.size(); ++j) {
            y[i] += A[i][j] * x[j];
        }
    }
    return y;
}

// output for matrix 

template <typename T>
void print_matrix(const matrix<T>& A) {

    int n = A.size();
    int m = A[0].size();
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < m; ++j){
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
}


int main(){
    
    std::cout << "Program: Gaussian Elimination" << std::endl;

    // set the output format for floating point numbers

    std::cout << std::scientific << std::setprecision(6);


    // define a custom float type to switch between single and double precision

    using float_type = double; // double or float


    // it is important to define constants of the correct type when switching 
    // between float and double, otherwise you may get unexpected results due 
    // to type promotion. For those problems where you use only double precision, 
    // you can use 1.0 and 0.0 (and other constants) directly

    float_type one = static_cast<float_type>(1.0);
    float_type zero = static_cast<float_type>(0.0);

    const int N{8};
    matrix<float_type> A = hilbert<float_type>(N);
    print_matrix(A);

    // exact solution and corresponding right-hand side

    vector<float_type> x_exact(N, one);
    vector<float_type> b = matvec(A, x_exact);


    // code for solving linear systems goes here

    // Newton's method for nonlinear system
    #include <cmath>
    auto f = [](const std::vector<double>& x) -> std::vector<double> {
        double x1 = x[0], x2 = x[1];
        double f1 = std::log(x1*x1 + x2*x2) - std::sin(x1*x2) - (std::log(2.0) + std::log(M_PI));
        double f2 = std::exp(x1 - x2) + std::cos(x1*x2);
        return {f1, f2};
    };

    auto jacobian = [](const std::vector<double>& x) -> matrix<double> {
        double x1 = x[0], x2 = x[1];
        double df1dx1 = 2*x1/(x1*x1 + x2*x2) - x2*std::cos(x1*x2);
        double df1dx2 = 2*x2/(x1*x1 + x2*x2) - x1*std::cos(x1*x2);
        double df2dx1 = std::exp(x1 - x2) - x2*std::sin(x1*x2);
        double df2dx2 = -std::exp(x1 - x2) - x1*std::sin(x1*x2);
        return {{df1dx1, df1dx2}, {df2dx1, df2dx2}};
    };

    // Solve J dx = -f(x) for dx (2x2 linear system)
    auto solve2x2 = [](const matrix<double>& J, const std::vector<double>& b) -> std::vector<double> {
        double a = J[0][0], b1 = J[0][1], c = J[1][0], d = J[1][1];
        double det = a*d - b1*c;
        if (std::abs(det) < 1e-12) throw std::runtime_error("Jacobian is singular");
        double dx1 = (d*(-b[0]) - b1*(-b[1])) / det;
        double dx2 = (a*(-b[1]) - c*(-b[0])) / det;
        return {dx1, dx2};
    };

    std::vector<double> x = {2.0, 2.0};
    int max_iter = 100;
    double tol = 1e-5;
    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> fx = f(x);
        double err = std::sqrt(fx[0]*fx[0] + fx[1]*fx[1]);
        if (err < tol) {
            std::cout << "Converged in " << iter << " iterations.\n";
            break;
        }
        matrix<double> J = jacobian(x);
        std::vector<double> dx = solve2x2(J, fx);
        x[0] += dx[0];
        x[1] += dx[1];
        std::cout << "Iter " << iter << ": x = [" << x[0] << ", " << x[1] << "], error = " << err << std::endl;
    }
    std::cout << "Solution: x = [" << x[0] << ", " << x[1] << "]" << std::endl;

}

