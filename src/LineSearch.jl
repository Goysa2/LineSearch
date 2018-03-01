module LineSearch
export ls_algorithms, interfaced_ls_algorithms

using Optimize, Polynomials
using LSDescentMethods
using LineSearches
# using PyPlot

include("includes.jl")

include("algorithms-ls.jl")

end # module
