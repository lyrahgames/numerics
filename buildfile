cxx.std = latest
using cxx

hxx{*}: extension = hpp
cxx{*}: extension = cpp

cxx.poptions =+ "-I$src_base"

./: exe{iterative_solvers}: cxx{iterative_solvers}
./: exe{poisson_equation_2d}: cxx{poisson_equation_2d}
./: exe{schroedinger_equation}: cxx{schroedinger_equation}