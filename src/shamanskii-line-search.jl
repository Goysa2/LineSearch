export shamanskii_line_search

"""
A line search algorithm for the globalized Shamanskii method described in

  Global Convergence Technique for the Newton
    Method with Periodic Hessian Evaluation

By F. LAMPARIELLO and M. SCIANDRONE

JOURNAL OF OPTIMIZATION THEORY AND APPLICATIONS:
Vol. 111, No. 2, pp. 341–358, November 2001
"""
function shamanskii_line_search(h           :: LineModel,
                                stop_ls     :: LS_Stopping,
                                f_meta      :: LS_Function_Meta;
                                δ           :: Float64 = 0.5,
                                verboseLS   :: Bool = false,
                                kwargs...)
    state = stop_ls.current_state
    i = 0
    update!(state, x = 1.0)

    stop_ls.optimality_check = (x, y) -> shamanskii_stop(x,y)

    # First try the unitary step size
    h1 = obj(h, state.x)
    slope1 = grad(h, state.x)

    OK = update_and_start!(stop_ls, ht = h1, gt = slope1, tmps = time())

    if OK
        printstyled("on ne fait aucune itération de line search \n", color = :red)
    end

    while !OK
        i += 1
        t = state.x
        update!(state, x = δ ^ i)
        ht = obj(h, state.x)
        slope = grad(h, state.x)
        OK = update_and_stop!(stop_ls, ht = ht, gt = slope)
        verboseLS && @printf(" iter  %4d  t  %7.1e slope %7.1e\n", stop_ls.meta.nb_of_stop, state.x, slope);
    end


    return (state, stop_ls.meta.optimal)
end # function
