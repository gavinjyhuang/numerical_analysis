#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

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

template<typename T>
std::vector<T> gaussian_elimination(std::vector<std::vector<T>> A, std::vector<T> b) {
    int n = A.size();
    for (int k = 0; k < n-1; ++k) {
        for (int i = k+1; i < n; ++i) {
            T m = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) {
                A[i][j] -= m * A[k][j];
            }
            b[i] -= m * b[k];
        }
    }
    std::vector<T> x(n);
    for (int i = n-1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i+1; j < n; ++j) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
    return x;
}

int main(){

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

    // exact solution and corresponding right-hand side

    vector<float_type> x_exact(N, one);
    vector<float_type> b = matvec(A, x_exact);


    std::cout << "Question 1b: Newton's Method" << std::endl;

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

    auto solve = [](const matrix<double>& J, const std::vector<double>& b) -> std::vector<double> {
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
        std::vector<double> dx = solve(J, fx);
        x[0] += dx[0];
        x[1] += dx[1];
        std::cout << "Iter " << iter << ": x = [" << x[0] << ", " << x[1] << "], error = " << err << std::endl;
    }
    std::cout << "Solution: x = [" << x[0] << ", " << x[1] << "]" << std::endl;

    std::cout << "\nQuestion 1d: Newton's Method" << std::endl;
    auto f3 = [](const std::vector<double>& x) -> std::vector<double> {
        double x1 = x[0], x2 = x[1], x3 = x[2];
        double f1 = 6*x1 - 2*std::cos(x2*x3) - 1;
        double f2 = 9*x2 + std::sqrt(x1*x1) + std::sin(x3) + 1.06 + 0.9;
        double f3 = 60*x3 + 3*std::exp(-x1*x2) + 10*M_PI - 3;
        return {f1, f2, f3};
    };

    auto jacobian3 = [](const std::vector<double>& x) -> matrix<double> {
        double x1 = x[0], x2 = x[1], x3 = x[2];
        // Partial derivatives
        double df1dx1 = 6;
        double df1dx2 = 2*x3*std::sin(x2*x3);
        double df1dx3 = 2*x2*std::sin(x2*x3);

        double df2dx1 = x1 == 0 ? 0 : x1/std::sqrt(x1*x1); 
        double df2dx2 = 9;
        double df2dx3 = std::cos(x3);

        double df3dx1 = -3*x2*std::exp(-x1*x2);
        double df3dx2 = -3*x1*std::exp(-x1*x2);
        double df3dx3 = 60;

        return {
            {df1dx1, df1dx2, df1dx3},
            {df2dx1, df2dx2, df2dx3},
            {df3dx1, df3dx2, df3dx3}
        };
    };

    auto solve2 = [](const matrix<double>& J, const std::vector<double>& b) -> std::vector<double> {
        double a11 = J[0][0], a12 = J[0][1], a13 = J[0][2];
        double a21 = J[1][0], a22 = J[1][1], a23 = J[1][2];
        double a31 = J[2][0], a32 = J[2][1], a33 = J[2][2];
        double det = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31);
        if (std::abs(det) < 1e-12) throw std::runtime_error("Jacobian is singular");
        // dx1
        double det1 = (-b[0])*(a22*a33 - a23*a32) - a12*((-b[1])*a33 - a23*(-b[2])) + a13*((-b[1])*a32 - a22*(-b[2]));
        // dx2
        double det2 = a11*((-b[1])*a33 - a23*(-b[2])) - (-b[0])*(a21*a33 - a23*a31) + a13*(a21*(-b[2]) - (-b[1])*a31);
        // dx3
        double det3 = a11*(a22*(-b[2]) - (-b[1])*a32) - a12*(a21*(-b[2]) - (-b[1])*a31) + (-b[0])*(a21*a32 - a22*a31);
        double dx1 = det1 / det;
        double dx2 = det2 / det;
        double dx3 = det3 / det;
        return {dx1, dx2, dx3};
    };

    std::vector<double> x3 = {0.0, 0.0, 0.0};
    int max_iter3 = 100;
    double tol3 = 1e-5;
    for (int iter = 0; iter < max_iter3; ++iter) {
        std::vector<double> fx = f3(x3);
        double err = std::sqrt(fx[0]*fx[0] + fx[1]*fx[1] + fx[2]*fx[2]);
        if (err < tol3) {
            std::cout << "Converged in " << iter << " iterations.\n";
            break;
        }
        matrix<double> J = jacobian3(x3);
        std::vector<double> dx = solve2(J, fx);
        x3[0] += dx[0];
        x3[1] += dx[1];
        x3[2] += dx[2];
        std::cout << "Iter " << iter << ": x = [" << x3[0] << ", " << x3[1] << ", " << x3[2] << "], error = " << err << std::endl;
    }
    std::cout << "Solution: x = [" << x3[0] << ", " << x3[1] << ", " << x3[2] << "]" << std::endl;
    
    
    std::cout << "\nQuestion 2: Pressure to sink object in soil" << std::endl;
    double r1 = 1 * 0.0254;
    double r2 = 2 * 0.0254;
    double r3 = 3 * 0.0254; 
    double p1 = 10 * 6894.76;
    double p2 = 12 * 6894.76;
    double p3 = 15 * 6894.76;

    // System:
    auto soil_f = [&](const std::vector<double>& k) -> std::vector<double> {
        double k1 = k[0], k2 = k[1], k3 = k[2];
        double f1 = k1 * std::exp(k2 * r1) + k3 * r1 - p1;
        double f2 = k1 * std::exp(k2 * r2) + k3 * r2 - p2;
        double f3 = k1 * std::exp(k2 * r3) + k3 * r3 - p3;
        return {f1, f2, f3};
    };

    auto soil_jacobian = [&](const std::vector<double>& k) -> matrix<double> {
        double k1 = k[0], k2 = k[1], k3 = k[2];
        // Partial derivatives
        return {
            {std::exp(k2 * r1), k1 * r1 * std::exp(k2 * r1), r1},
            {std::exp(k2 * r2), k1 * r2 * std::exp(k2 * r2), r2},
            {std::exp(k2 * r3), k1 * r3 * std::exp(k2 * r3), r3}
        };
    };

    std::vector<double> k = {10000.0, 10.0, 10000.0};
    int max_iter_soil = 50;
    double tol_soil = 1e-5;
    for (int iter = 0; iter < max_iter_soil; ++iter) {
        std::vector<double> fk = soil_f(k);
        double err = std::sqrt(fk[0]*fk[0] + fk[1]*fk[1] + fk[2]*fk[2]);
        if (err < tol_soil) {
            std::cout << "Converged for soil constants in " << iter << " iterations.\n";
            break;
        }
        matrix<double> J = soil_jacobian(k);
        double a11 = J[0][0], a12 = J[0][1], a13 = J[0][2];
        double a21 = J[1][0], a22 = J[1][1], a23 = J[1][2];
        double a31 = J[2][0], a32 = J[2][1], a33 = J[2][2];
        double det = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31);
        if (std::abs(det) < 1e-12) throw std::runtime_error("Jacobian is singular");

        double det1 = (-fk[0])*(a22*a33 - a23*a32) - a12*((-fk[1])*a33 - a23*(-fk[2])) + a13*((-fk[1])*a32 - a22*(-fk[2]));

        double det2 = a11*((-fk[1])*a33 - a23*(-fk[2])) - (-fk[0])*(a21*a33 - a23*a31) + a13*(a21*(-fk[2]) - (-fk[1])*a31);
        double det3 = a11*(a22*(-fk[2]) - (-fk[1])*a32) - a12*(a21*(-fk[2]) - (-fk[1])*a31) + (-fk[0])*(a21*a32 - a22*a31);
        double dx1 = det1 / det;
        double dx2 = det2 / det;
        double dx3 = det3 / det;
        k[0] += dx1;
        k[1] += dx2;
        k[2] += dx3;
        std::cout << "Iter " << iter << ": k = [" << k[0] << ", " << k[1] << ", " << k[2] << "], error = " << err << std::endl;
    }
    std::cout << "Soil constants: k1 = " << k[0] << ", k2 = " << k[1] << ", k3 = " << k[2] << std::endl;

    // Part b: Find minimal radius for 500 lb load, sinkage < 1 ft

    double F = 500 * 4.44822; 
    double max_sinkage = 1 * 0.3048; 
    auto pressure = [&](double r) -> double {
        return k[0] * std::exp(k[1] * r) + k[2] * r;
    };
    double required_p = F / (M_PI * std::pow(r1, 2));
    double min_r = 0.01;
    double found_r = -1;
    for (double r = min_r; r < 0.5; r += 0.001) {
        double p = pressure(r);
        double area = M_PI * r * r;
        double p_required = F / area;
        if (p >= p_required) {
            found_r = r;
            std::cout << "Minimal plate radius for 500 lb load and <1ft sinkage: " << r << " m (" << r/0.0254 << " in)" << std::endl;
            break;
        }
    }

    std::cout << "\nQuestion 3: Population dymanics for 3 competing species" << std::endl;
    double alpha = 0.5, beta = 0.25;
    auto F_pop = [&](const std::vector<double>& x) -> std::vector<double> {
        double x1 = x[0], x2 = x[1], x3 = x[2];
        double f1 = x1 * (1 - x1 - alpha * x2 - beta * x3);
        double f2 = x2 * (1 - x2 - beta * x1 - alpha * x3);
        double f3 = x3 * (1 - x3 - alpha * x1 - beta * x2);
        return {f1, f2, f3};
    };

    // Broyden's method 
    std::vector<double> x_pop = {0.5, 0.5, 0.5};
    matrix<double> B = {{1,0,0},{0,1,0},{0,0,1}};
    int max_iter2 = 100;
    double tol2 = 1e-8;
    for (int iter = 0; iter < max_iter2; ++iter) {
        std::vector<double> Fx = F_pop(x_pop);
        double err = std::sqrt(Fx[0]*Fx[0] + Fx[1]*Fx[1] + Fx[2]*Fx[2]);
        if (err < tol2) {
            std::cout << "Converged in " << iter << " iterations.\n";
            break;
        }
        double a11 = B[0][0], a12 = B[0][1], a13 = B[0][2];
        double a21 = B[1][0], a22 = B[1][1], a23 = B[1][2];
        double a31 = B[2][0], a32 = B[2][1], a33 = B[2][2];
        double det = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31);
        if (std::abs(det) < 1e-12) throw std::runtime_error("Jacobian is singular");
        double det1 = (-Fx[0])*(a22*a33 - a23*a32) - a12*((-Fx[1])*a33 - a23*(-Fx[2])) + a13*((-Fx[1])*a32 - a22*(-Fx[2]));
        double det2 = a11*((-Fx[1])*a33 - a23*(-Fx[2])) - (-Fx[0])*(a21*a33 - a23*a31) + a13*(a21*(-Fx[2]) - (-Fx[1])*a31);
        double det3 = a11*(a22*(-Fx[2]) - (-Fx[1])*a32) - a12*(a21*(-Fx[2]) - (-Fx[1])*a31) + (-Fx[0])*(a21*a32 - a22*a31);
        std::vector<double> dx = {det1/det, det2/det, det3/det};
        std::vector<double> x_new = {x_pop[0]+dx[0], x_pop[1]+dx[1], x_pop[2]+dx[2]};

        x_new[0] = std::max(0.0, std::min(1.0, x_new[0]));
        x_new[1] = std::max(0.25, std::min(1.0, x_new[1]));
        x_new[2] = std::max(0.25, std::min(1.0, x_new[2]));
        std::vector<double> Fx_new = F_pop(x_new);
        std::vector<double> y = {Fx_new[0]-Fx[0], Fx_new[1]-Fx[1], Fx_new[2]-Fx[2]};
        std::vector<double> s = {x_new[0]-x_pop[0], x_new[1]-x_pop[1], x_new[2]-x_pop[2]};
        
        std::vector<double> Bs = {
            B[0][0]*s[0]+B[0][1]*s[1]+B[0][2]*s[2],
            B[1][0]*s[0]+B[1][1]*s[1]+B[1][2]*s[2],
            B[2][0]*s[0]+B[2][1]*s[1]+B[2][2]*s[2]
        };
        std::vector<double> y_minus_Bs = {y[0]-Bs[0], y[1]-Bs[1], y[2]-Bs[2]};
        double sTs = s[0]*s[0]+s[1]*s[1]+s[2]*s[2];
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) B[i][j] += y_minus_Bs[i]*s[j]/(sTs+1e-12);
        x_pop = x_new;
        std::cout << "Iter " << iter << ": x = [" << x_pop[0] << ", " << x_pop[1] << ", " << x_pop[2] << "], error = " << err << std::endl;
    }
    std::cout << "Stable equilibrium: x = [" << x_pop[0] << ", " << x_pop[1] << ", " << x_pop[2] << "]" << std::endl;




    std::cout << "\nQuestion 6: implement gaussian elim without pivoting" << std::endl;

    matrix<double> H3 = hilbert<double>(3);
    std::cout << "Original 3x3 Hilbert matrix:" << std::endl;
    print_matrix(H3);

    std::vector<double> b3 = {1.0, 1.0, 1.0};
    matrix<double> U3 = H3;

    int n = 3;
    for (int k = 0; k < n-1; ++k) {
        for (int i = k+1; i < n; ++i) {
            double m = U3[i][k] / U3[k][k];
            for (int j = k; j < n; ++j) {
                U3[i][j] -= m * U3[k][j];
            }
        }
    }
    std::cout << "Upper triangular matrix after elimination:" << std::endl;
    print_matrix(U3);

    std::cout << "\nQuestion 7: solve 8x8 hilbert" << std::endl;

    auto relative_error = [](const std::vector<double>& x, const std::vector<double>& x_exact) {
        double num = 0, denom = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            num += (x[i] - x_exact[i]) * (x[i] - x_exact[i]);
            denom += x_exact[i] * x_exact[i];
        }
        return std::sqrt(num) / std::sqrt(denom);
    };

    // Single precision
    {
        matrix<float> H8f = hilbert<float>(8);
        std::vector<float> x_exact(8, 1.0f);
        std::vector<float> b = matvec(H8f, x_exact);
        std::vector<float> x = gaussian_elimination(H8f, b);
        // Convert to double for error calculation
        std::vector<double> x_d(x.begin(), x.end()), x_exact_d(x_exact.begin(), x_exact.end());
        double rel_err = relative_error(x_d, x_exact_d);
        std::cout << "Single precision relative error: " << rel_err << std::endl;
    }

    // Double precision
    {
        matrix<double> H8d = hilbert<double>(8);
        std::vector<double> x_exact(8, 1.0);
        std::vector<double> b = matvec(H8d, x_exact);
        std::vector<double> x = gaussian_elimination(H8d, b);
        double rel_err = relative_error(x, x_exact);
        std::cout << "Double precision relative error: " << rel_err << std::endl;
    }


    std::cout << "\nQuestion 8: Rodent population model" << std::endl;
    // Fecundity rates for ages 1, 2, 3
    std::vector<double> F_vec = {0.5, 1.7, 0.7};
    // Survival rates 
    std::vector<double> S8 = {0.6, 0.5};
    // Year 2 population
    std::vector<double> N2_8 = {75, 18, 10};

    // (a) Population in year 1
    double N1_1 = N2_8[1] / S8[0];
    double N1_2 = N2_8[2] / S8[1];

    double N1_3 = (N2_8[0] - F_vec[0]*N1_1 - F_vec[1]*N1_2) / F_vec[2];
    std::cout << "Year 1 population: N1 = [" << N1_1 << ", " << N1_2 << ", " << N1_3 << "]" << std::endl;

    // (b) Population in year 3
    std::vector<double> N3_8(3);
    N3_8[0] = F_vec[0]*N2_8[0] + F_vec[1]*N2_8[1] + F_vec[2]*N2_8[2];
    N3_8[1] = S8[0]*N2_8[0];
    N3_8[2] = S8[1]*N2_8[1];
    std::cout << "Year 3 population: N3 = [" << N3_8[0] << ", " << N3_8[1] << ", " << N3_8[2] << "]" << std::endl;

}

