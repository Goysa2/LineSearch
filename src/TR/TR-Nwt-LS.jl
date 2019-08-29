export TR_Nwt_ls

"""
A one dimensionnal trust region algorithm using the exact second derivative in
the second order Taylor expansion. For more documentation referer to TR_generic_ls.
"""
function TR_Nwt_ls(h :: LineModel,  stop_ls :: LS_Stopping,
                  f_meta  :: LS_Function_Meta;  kwargs...)

    (state, stop_ls.meta.optimal) = TR_generic_ls(h, stop_ls, f_meta; kwargs...)
    return (state, stop_ls.meta.optimal)
end
