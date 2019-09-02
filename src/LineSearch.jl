module LineSearch

using Printf

using State
using Stopping
using NLPModels
using NLPModelsJuMP
using Printf
using LinearAlgebra
using OptimizationProblems

include("LSFunctionMetamod.jl")
include("LineModel2mod.jl")

include("phi_dphi.jl")
include("pred_ared.jl")
include("update_H.jl")

include("armijo_backtracking.jl")
include("one-step-size.jl")
include("shamanskii-line-search.jl")

include("TR/TR_generic_ls.jl")
include("TR/TR-ls-step-computation.jl")
include("TR/TR-Nwt-LS.jl")
include("TR/TR-Sec-LS.jl")
include("TR/TR-SecA-LS.jl")

include("ARC/ARC_generic_ls.jl")
include("ARC/ARC-Nwt-LS.jl")
include("ARC/ARC-Sec-LS.jl")
include("ARC/ARC-SecA-LS.jl")
include("ARC/ARC_direction_computation.jl")

include("Bisection/find_interval.jl")
include("Bisection/Biss-Nwt.jl")
include("Bisection/Biss-Sec.jl")
include("Bisection/Biss-SecA.jl")
include("Bisection/Biss-Cub.jl")

include("other-ls/struct-ls.jl")
include("other-ls/hagerzhang2.jl")
include("other-ls/hagerzhang.jl")

end # module
