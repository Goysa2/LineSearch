module LineSearch
export ls_algorithms, interfaced_ls_algorithms

# using Optimize, Polynomials
using LineSearches
using LSDescentMethods

# using PyPlot

include("includes.jl")

include("algorithms-ls.jl")

end # module
