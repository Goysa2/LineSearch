using LSDescentMethods
using LineSearches
using NLPModels, Optimize

include("Armijo.jl")
include("Wolfe.jl")
include("Nwt_computation_ls.jl")
include("Sec_computation_ls.jl")
include("SecA_computation_ls.jl")
include("stopping_ls.jl")


include("ARC/ARC-Cub-LS.jl")
include("ARC/ARC-Nwt-LS.jl")
include("ARC/ARC-Sec-LS.jl")
include("ARC/ARC-SecA-LS.jl")

include("ARC/init_ARC.jl")
include("ARC/ARC_generic_ls.jl")

include("Bissection/trouve-intervalle-ls.jl")
include("Bissection/Biss-LS.jl")
include("Bissection/Biss-Cub-LS.jl")
include("Bissection/Biss-Nwt-LS.jl")
include("Bissection/Biss-Sec-LS.jl")
include("Bissection/Biss-SecA-LS.jl")

include("TR/TR-Cub-LS.jl")
include("TR/TR-Nwt-LS.jl")
include("TR/TR-Sec-LS.jl")
include("TR/TR-SecA-LS.jl")

include("TR/init-TR.jl")
include("TR/TR_generic_ls.jl")

include("TR/TR-ls-step-computation.jl")

include("zoom/trouve-intervalleA-ls.jl")
include("zoom/zoom-ls.jl")
include("zoom/zoom-Nwt-ls.jl")
include("zoom/zoom-Cub-ls.jl")
include("zoom/zoom-Sec-ls.jl")
include("zoom/zoom-SecA-ls.jl")

include("zoom/zoom-generic-ls.jl")
include("zoom_computation.jl")

include("Other-LS/strongwolfe2.jl")
include("Other-LS/morethuente2.jl")
include("Other-LS/hagerzhang2.jl")
include("Other-LS/backtracking2.jl")

#graphic tool
include("graph_linefunc.jl")
