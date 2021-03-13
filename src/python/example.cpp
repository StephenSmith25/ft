#include "pybind11/pybind11.h"
#include <string>
namespace py = pybind11;

int add(int i, int j)
{
  return i + j;
}


// creating bindings for a custom type
struct Pet
{
  explicit Pet(std::string name) : name(std::move(name)) {}
  void setName(const std::string &name_) { name = name_; }
  const std::string &getName() const { return name; }
  std::string name;
};


PYBIND11_MODULE(example, m)
{
  py::class_<Pet>(m, "Pet")
    .def(py::init<const std::string &>())
    .def_property("name", &Pet::getName, &Pet::setName);
}
