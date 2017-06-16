module LineSearch
export algorithms, interfaced_algorithms
export Newton_linesearch

using JuMP, NLPModels, Optimize
using Polynomials
using ScalarOptimizationProblems
using ScalarSolvers

using LSDescentMethods

using LineSearches

using Plots, PyPlot

include("includes.jl")

include("algorithms.jl")

end # module
