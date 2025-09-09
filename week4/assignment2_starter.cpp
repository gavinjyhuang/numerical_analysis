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

}

