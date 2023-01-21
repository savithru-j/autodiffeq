#include <autodiffeq/numerics/ADVar.hpp>
#include <iostream>

using namespace autodiffeq;

int main()
{

  ADVar<double> x(1.0, {0.0, 1.0});
  std::cout << x.value() << std::endl;


}