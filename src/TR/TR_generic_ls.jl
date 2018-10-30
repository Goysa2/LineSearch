export TR_generic_ls
"""
A generic one dimensionnal Trust Region algorithm adapted for line search.
In this generic version we assume a second order polynomial is used to
approximate our h function. A third degree approximation is used in TR_Cub_ls
We use q(d) = h(θ) + h'(θ)⋆d + 0.5 * h''(θ) ⋆ d². Also h''(θ) is sometimes
approximated by a secant (Sec) or an improved secant (SecA). That approximation
is decided with hyper parameter "direction"

Some specific hyper parameter to this line search:
symmetrical :: The method is designed to used an asymmetrical  trust region
to avoid having a negative step size. If symmetrical true then the trust region
becomes symmetrical

Some parameters are used for TR and ARC methods
eps1 and eps2 :: treshold to determine if q is a bad or a good++ approximation
of h
red and aug :: Controls the speed at which we reduce or augment the size of
the trust region depending on the previously mentionned treshold.

All the parameters mentionned above can be found in the LS_Function_Meta
"""
function TR_generic_ls(h           :: LineModel,
                       stop_ls     :: LS_Stopping,
                       f_meta      :: LS_Function_Meta;
                       φ_dφ        :: Function = (x, y) -> phi_dphi(x, y),
                       verboseLS   :: Bool = false,
                       symmetrical :: Bool = false,
                       kwargs...)



  #state = Array{LS_Stopping.current_state}(0) # Array contenant les states
                                               # des itérations précédentes
  state = stop_ls.current_state

  tmp = obj(h, 1.0); tmpg = grad(h, 1.0)
  # # Check if 1.0 is an admissible step size.
  # OK = start!(stop_ls)
  OK = update_and_start!(stop_ls, ht = tmp, gt = tmpg, tmps = time())

  # If 1.0 isn't an admissible step size we start our line search algorithm
  iter = 0

  # If 1.0 is a good step size we don't want to compute  unnecessary function
  # evaluations
  H = NaN; φt = NaN; dφt = NaN                   # only for the verbose

  #if OK == false
    φt, dφt = φ_dφ(h, state) #Tangi: Ici on recalcule la fct objectif ?

    tprec = NaN; φtestTR = NaN; dφtestTR = NaN; φtprec = NaN; dφtprec = NaN;

    # H will denote the approximation to φ'' hereafter
    if f_meta.dir == "Nwt"
      H = hess(h, state.x)
    elseif f_meta.dir == "Sec" || f_meta.dir == "SecA"
      H = 1.0
    end
  #end # if OK == false

  # Quadratic approximation of h
  # q(d) = φt + dφt * d + 0.5 * H * d^2

  verboseLS &&
    @printf("  iter   t       φt        dφt        ddφt        Δn         Δp")
  verboseLS &&
    @printf("   Successful        dN        ratio\n")
  verboseLS &&
    @printf("%4d %9.2e  %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter, state.x, φt, dφt, H,
                                                          f_meta.Δn, f_meta.Δp);

  while !OK #admissible: Armijo & Wolfe, tired: iterations
      # We select our descent direction. We compute the classical Newton
      # direction and then we check we the bounds of our trust region
      dN = -dφt/H
      d = TR_ls_step_computation(H, dφt, dN, f_meta.Δn, f_meta.Δp)

      # We see where the direction d gets us
      candidate_state = copy(state)
      update!(candidate_state, x = state.x + d)
      φtestTR, dφtestTR = φ_dφ(h, candidate_state)

      # depending on the approximation of the second derivate we need different
      # information
      if f_meta.dir == "Sec" || f_meta.dir == "SecA"
        tprec = state.x
        dφtprec = dφt
      end
      if f_meta.dir == "SecA" φtprec = φt end

      # We compute the predicted and actual reduction as well as the ratio
      # between the two.
      (pred, ared, ratio) = pred_ared_computation(dφt, φt, H, d, φtestTR,
                                                  dφtestTR)
      # mettre le if qui suit dans une fonction pour aléger le code.
      if ratio < f_meta.eps1  # Unsuccessful approximation => reduce interval
        f_meta.Δp = f_meta.red * f_meta.Δp
        f_meta.Δn = f_meta.red * f_meta.Δn
        iter+=1
        verboseLS && @printf("%4d %9.2e %9.2e  %9.2e %9.2e %9.2e %9.2e %9e %9.2e %9e\n",
                              iter, state.x, φt, dφt, H, f_meta.Δn, f_meta.Δp, 0, dN, ratio);
      else             # Successful approximation => we move towards an
                       # admissible step size
        stop_ls.current_state = copy(candidate_state)
        state = stop_ls.current_state
        φt, dφt = φtestTR, dφtestTR
        H = update_H(f_meta.dir, h, state.x, tprec, φt, dφt, φtprec, dφtprec)

        if symmetrical        # We adjust our interval if we want it symmetrical
                              # or not
          if ratio > f_meta.eps2 # Very Successful iteration => augment size of
                                 # trust region
            f_meta.Δp = f_meta.aug * f_meta.Δp
            f_meta.Δn = f_meta.aug * f_meta.Δn
          end
        else
          if ratio > f_meta.eps2  # Very Successful iteration=> augment size of
                                  # trust region
            f_meta.Δp = f_meta.aug * f_meta.Δp
            f_meta.Δn = min(-state.x, f_meta.Δn)
          else
            f_meta.Δn = min(-state.x, f_meta.Δn)
          end
        end # à mettre dans une fonction

          iter += 1 # à mettre dans ls_at_t a un certain point

          verboseLS &&
            @printf("%4d %9.2e %9.2e  %9.2e %9.2e %9.2e %9.2e %9e %9.2e  %9e\n",
                    iter, state.x, φt, dφt, H, f_meta.Δn, f_meta.Δp, 1, dN, ratio);
      end;
      OK = stop!(stop_ls)
  end;

  return (state, stop_ls.meta.optimal)

end # function
