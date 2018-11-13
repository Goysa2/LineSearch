module LineSearch

using Printf

using State
using Stopping
using NLPModels

include("LSFunctionMetamod.jl")
include("LineModel2mod.jl")

include("phi_dphi.jl")
include("pred_ared.jl")
include("update_H.jl")

include("armijo_backtracking.jl")
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


end # module
