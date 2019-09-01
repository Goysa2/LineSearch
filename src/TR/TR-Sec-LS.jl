export TR_Sec_ls
"""
A one dimensionnal trust region algorithm using the secant as the approximation
for the second derivative in the second order Taylor expansion.
For more documentation referer to TR_generic_ls.
"""
function TR_Sec_ls(h :: LineModel,  stop_ls :: LS_Stopping;  kwargs...)
    ls_meta = LS_Function_Meta(dir = :Sec)
    (state, stop_ls.meta.optimal) = TR_generic_ls(h, stop_ls; f_meta = ls_meta, kwargs...)
    return (state, stop_ls.meta.optimal)
end
