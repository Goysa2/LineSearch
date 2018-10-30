export armijo_ls

"""
An armijo backtracking algorithm.
"""
function armijo_ls(h           :: LineModel,
                   stop_ls     :: LS_Stopping,
                   f_meta      :: LS_Function_Meta;
                   φ_dφ        :: Function = (x, y) -> phi_dphi(x, y),
                   verboseLS   :: Bool = false,
                   kwargs...)
    state = stop_ls.current_state
    nbk = 0
    nbW = 0
    update!(state, x = 1.0)

    stop_ls.optimality_check = (x, y) -> armijo(x,y)

    # First try to increase t
    h1 = obj(h, state.x)
    slope1 = grad(h, state.x)

    OK = update_and_start!(stop_ls, ht = h1, gt = slope1, tmps = time())

    while !OK
        t = state.x
        update!(state, x = 0.5 * t)
        ht = obj(h, state.x)
        slope = grad(h, state.x)
        OK = update_and_stop!(stop_ls, ht = ht, gt = slope)
        verboseLS && @printf(" iter  %4d  t  %7.1e slope %7.1e\n", stop_ls.meta.nb_of_stop, state.x, slope);
    end


    return (state, stop_ls.meta.optimal)
end # function
