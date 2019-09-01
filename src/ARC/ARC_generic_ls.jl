export ARC_generic_ls

"""
A one dimensionnal ARC algorithm designed for Line Search purposes.

In this generic version we assume a second order polynomial enriched with a cubic
regularization term is used to approximate our h function.
We use c(d) = h(θ) + h'(θ)⋆d + 0.5 * h''(θ) ⋆ d² + Δ/3 |d|³.
Also h''(θ) is sometimes approximated by a secant (Sec) or an improved secant (SecA).
That approximation
is decided with hyper parameter "direction"


Some specific parameters for theses algorithms:
  - Δ :: The cubic regularization parameter
Some parameters are used for TR and ARC methods:
  -eps1 and eps2 :: treshold to determine if q is a bad or a good++ approximation
    of h
  -red and aug :: Controls the speed at which we reduce or augment the size of
    the trust region depending on the previously mentionned treshold.

All the parameters mentionned above can be found in the LS_Function_Meta
"""
function ARC_generic_ls(h         :: LineModel, stop_ls   :: LS_Stopping;
                        f_meta = LS_Function_Meta(),
                        φ_dφ      :: Function = (x, y) -> phi_dphi(x, y),
                        verboseLS :: Bool = false, kwargs...)

    state = stop_ls.current_state

    update!(state, x = 1.0, ht = obj(h, 1.0), gt = grad(h, 1.0), tmps = time())

    # Check if 1.0 is an admissible step size.
    OK = start!(stop_ls)

    # If 1.0 isn't an admissible step size we start our line search algorithm
    iter = 0
    # If 1.0 is a good step size we don't want to compute  unnecessary function
    # evaluations
    H = NaN; φt = NaN; dφt = NaN                   # only for the verbose
    if OK == false
      φt, dφt = φ_dφ(h, state)

      tprec = NaN; φtestTR = NaN; dφtestTR = NaN; φtprec = NaN; dφtprec = NaN;

      # H will denote the approximation to φ'' hereafter
      if f_meta.dir == :Nwt
        H = hess(h, state.x)
      elseif f_meta.dir == :Sec || f_meta.dir == :SecA
        H = 1.0
      end
    end # if OK == false

    verboseLS && @printf("   iter   t       φt        dφt        Δ")
    verboseLS && @printf("   Successful        d        ratio\n")
    verboseLS && @printf("%4d %9.2e  %9.2e  %9.2e %9.2e\n",
                         iter, state.x, φt, dφt, f_meta.Δ);

    while !OK # admissible: respecte armijo et wolfe,
                                # tired: nb d'itérations

        d = ARC_step_computation(H, dφt, f_meta.Δ; kwargs...)

        # We see where the direction d gets us
        candidate_state = copy(state)
        update!(candidate_state, x = state.x + d)
        φtestTR, dφtestTR = φ_dφ(h, candidate_state)

        # depending on the approximation of the second derivate we need different
        # information
        if f_meta.dir == :Sec || f_meta.dir == :SecA
          tprec = state.x
          dφtprec = dφt
        end
        if f_meta.dir == :SecA φtprec = φt end

        # test d'arrêt sur dφ
        (pred, ared, ratio) =
               pred_ared_computation(dφt, φt, H, d, φtestTR, dφtestTR)

        if (state.x + d < 0.0) || ratio < f_meta.eps1  # Unsuccessful
          f_meta.Δ = f_meta.red * f_meta.Δ
          iter += 1
          verboseLS &&
            @printf("%4d %9.2e %9.2e  %9.2e  %9.2e %9.2e %7.2e\n",
                    iter, state.x, φt, dφt, f_meta.Δ, 0, d);
        else             # Successful
            stop_ls.current_state = copy(candidate_state)
            state = stop_ls.current_state
            φt, dφt = φtestTR, dφtestTR
            H = update_H(f_meta.dir, h, state.x, tprec, φt, dφt, φtprec, dφtprec)

          if ratio > f_meta.eps2
            f_meta.Δ = f_meta.aug * f_meta.Δ
          end
          iter += 1
          verboseLS &&
            @printf("%4d %9.2e %9.2e  %9.2e  %9.2e  %9.2e %7.2e \n",
                    iter, state.x, φt, dφt, f_meta.Δ, 1, d);
        end;
        OK = stop!(stop_ls)
    end;

    return (state, stop_ls.meta.optimal)

end
