module LineSearch
export algo_biss,problem_test,Intervals_probs,zoom_methods

using JuMP
using NLPModels
using Optimize
using Roots
using ScalarOptimizationProblems

include("includes.jl")

include("algorithms.jl")

algo_biss=[Biss_ls,Biss_Cub_ls,Biss_Nwt_ls,Biss_Sec_ls,Biss_SecA_ls]
zoom_methods=[zoom_ls,zoom_Cub_ls,zoom_nwt_ls,zoom_sec_ls,zoom_secA_ls]
problem_test=[AMPGO02(),AMPGO11()]
Intervals_probs=[2.7 7.5;-float(pi) 2*float(pi)]
end # module
