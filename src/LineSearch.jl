module LineSearch
export algorithms, interfaced_algorithms
export Newton_linesearch

using JuMP, NLPModels, Optimize
using PolynomialRoots
using ScalarOptimizationProblems
using ScalarSolvers

using LSDescentMethods

include("includes.jl")

using LineSearches

using Plots, PyPlot

# include("includes.jl")

include("algorithms.jl")

end # module
