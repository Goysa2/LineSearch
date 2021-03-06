export ARC_SecA_ls
"""
A one dimensionnal ARC algorithm using the exact second derivative in
the second order Taylor expansion enriched with a cubic term.
For more documentation referer to ARC_generic_ls.
"""
function ARC_SecA_ls(h :: LineModel, stop_ls :: LS_Stopping; kwargs...)
    ls_meta = LS_Function_Meta(dir = :Sec)
    (state, stop_ls.meta.optimal) = ARC_generic_ls(h, stop_ls, f_meta = ls_meta; kwargs...)

    return (state, stop_ls.meta.optimal)
end
