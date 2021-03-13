#include "pybind11/pybind11.h"


int add(int i, int j)
{
  return i + j;
}


PYBIND11_MODULE(example, m)
{
  m.doc() = "pybind example plugin";

  m.def("add", &add, "A Function with two numbers");
}