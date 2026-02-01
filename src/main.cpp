#include<nanobind/nanobind.h>

namespace nb = nanobind;

// Module definition
NB_MODULE(core, m) {
    m.doc() = "Sanctimonia C++ core module";
    m.def("ping", []() { return "pong"; }, "A simple test function");
}