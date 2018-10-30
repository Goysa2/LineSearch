export TR_SecA_ls
"""
A one dimensionnal trust region algorithm using the improved secant as the
approximation for the second derivative in the second order Taylor expansion.
For more documentation referer to TR_generic_ls.
"""
function TR_SecA_ls(h :: LineModel,
                   stop_ls :: LS_Stopping,
                   f_meta :: LS_Function_Meta;
                   kwargs...)
    f_meta.dir = "SecA"
    (state, stop_ls.meta.optimal) = TR_generic_ls(h, stop_ls, f_meta; kwargs...)
    return (state, stop_ls.meta.optimal)
end
