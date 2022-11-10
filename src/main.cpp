#include "nmbuflowtorch/tmp.hpp"
#include "Eigen/Dense"
#include <iostream>
#include <ostream>

using namespace std;
using namespace tmp;
using Eigen::MatrixXd;



int main(int argc, char **argv) {
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;

    std::cout << tmp::add(1, 2) << std::endl;
    return 0;
}