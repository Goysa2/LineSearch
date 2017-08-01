module LineSearch
export ls_algorithms, interfaced_ls_algorithms

using Optimize, PolynomialRoots
using LSDescentMethods
using LineSearches
using PyPlot

include("includes.jl")

include("algorithms-ls.jl")

end # module
