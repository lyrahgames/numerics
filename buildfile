cxx.std = latest
using cxx

hxx{*}: extension = hpp
cxx{*}: extension = cpp

cxx.poptions =+ "-I$src_base"

./: exe{poisson_equation_2d}: cxx{poisson_equation_2d}