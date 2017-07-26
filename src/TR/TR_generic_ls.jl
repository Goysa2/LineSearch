export TR_generic_ls

# A generic one dimensionnal Trust Region algorithm adapted for line search.
# In this generic version we assume a second order polynomial is used to
# approximate our h function. A third degree approximation is used in TR_Cub_ls
# We use q(d) = h(θ) + h'(θ)⋆d + 0.5 * h''(θ) ⋆ d². Also h''(θ) is sometimes
# approximated by a secant (Sec) or an improved secant (SecA). That approximation
# is decided with hyper parameter "direction"

# Some specific hyper parameter to this line search:
# symmetrical :: The method is designed to used an asymmetrical  trust region
# to avoid having a negative step size. If symmetrical true then the trust region
# becomes symmetrical

# Some parameters are used for TR and ARC methods
# eps1 and eps2 :: treshold to determine if q is a bad or a good++ approximation
# of h
# red and aug :: Controls the speed at which we reduce or augment the size of
# the trust region depending on the previously mentionned treshold.

function TR_generic_ls(h :: LineModel,
                       h₀ :: Float64,
                       g₀ :: Float64,
                       g :: Array{Float64,1};
                       stp_ls :: TStopping_LS = TStopping_LS(),
                       τ₀ :: Float64 = 0.0001,
                       τ₁ :: Float64 = 0.9999,
                       verboseLS :: Bool=false,
                       symmetrical :: Bool = false,
                       direction :: String="Nwt",
                       check_param :: Bool = false,
                       debug :: Bool = false,
                       add_step :: Bool = false,
                       check_slope :: Bool = false,
                       kwargs...)

    # We can check if the inputs are correct
    (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))
    if check_slope
      (abs(g₀ - grad(h, 0.0)) < 1e-4) || warn("wrong slope")
      verboseLS && @show h₀ obj(h, 0.0) g₀ grad(h,0.0)
    end

    start_ls!(g, stp_ls, τ₀, τ₁, h₀, g₀; kwargs...)

    # We check if 1.0 is an admissible step size
    global t = 1.0; t_original = Base.NaN
    (t,ht,gt,A_W,Δp,Δn)=init_TR(h,h₀,g₀,g,τ₀,τ₁;kwargs...)

    #If 1.0 is an admissible step size we don't need to go furter
    if A_W
      return (t,t_original,true,ht,0.0,0.0,false)
    end

    # If 1.0 isn't an admissible step size we start our line search algorithm
    iter = 0

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # function and
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # derivative

    tprec = NaN; φtestTR = NaN; dφtestTR = NaN; φtprec = NaN; dφtprec = NaN;

    # The rest of the algorithm work with φ
    # therefore, Armijo condition will be satisfied when φ(t)<φ(0)=0
    φt = ht - h₀ - τ₀*t*g₀
    dφt = gt - τ₀*g₀
    if t == 0.0
      φt = 0.0              # known that φ(0) = 0.0
      dφt = (1.0 - τ₀)*g₀   # known that φ'(0) = (1.0 - τ₀) * h'(0)
    end

    # H will denote the approximation to φ'' hereafter
    if direction == "Nwt"
      H = hess(h,t)
    elseif direction == "Sec" || direction == "SecA"
      H = 1.0
    end

    # Quadratic approximation of h
    q(d) = φt + dφt * d + 0.5 * H * d^2

    verboseLS && println("ϵₐ = $(stp_ls.ɛa) ϵᵦ = $(stp_ls.ɛb)")

    # Stopping criterion of the algorithm
    admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)

    debug && PyPlot.figure(1)
    debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])

    verboseLS && @printf("  iter   t       φt        dφt        ddφt        Δn        Δp        Successful        dN        ratio\n")
    verboseLS && @printf("%4d %9.2e  %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter,t,φt,dφt,ddφt,Δn,Δp);

    while !(admissible | tired) #admissible: Armijo & Wolfe, tired: iterations
        # We select our descent direction. We compute the classical Newton
        # direction and then we check we the bounds of our trust region
        dN = -dφt/H
        d = TR_ls_step_computation(H, dφt, dN, Δn, Δp)

        # We see where the direction d get us
        φtestTR = φ(t+d)
        dφtestTR= dφ(t+d)

        # depending on the approximation of the second derivate we need different
        # information
        if direction=="Sec"
          tprec = t
          dφtprec = dφt
        elseif direction=="SecA"
          tprec = t
          φtprec = φt
          dφtprec = dφt
        end

        # We compute the predicted and actual reduction as well as the ration
        # between the two.
        (pred, ared, ratio) = pred_ared_computation(dφt, φt, H, d, φtestTR, dφtestTR)

        if ratio < stp_ls.eps1  # Unsuccessful approximation => reduce the interval
            Δp = stp_ls.red*Δp
            Δn = stp_ls.red*Δn
            iter+=1
            verboseLS && @printf("%4d %9.2e %9.2e  %9.2e %9.2e %9.2e %9.2e %9e %9.2e %9e\n", iter,t,φt,dφt,ddφt,Δn,Δp,0,dN,ratio);
        else             # Successful approximation => we move towards a admissible
                         # step size

            # depending on our approximation we adjust our parameters
            # if direction=="Nwt"
            #   (t, φt, dφt, H) = Nwt_computation_ls(t, d, φtestTR, dφtestTR, h)
            # elseif direction=="Sec"
            #   (t, φt, dφt, H) = Sec_computation_ls(t, tprec, dφtprec, d, φtestTR, dφtestTR)
            # elseif direction=="SecA"
            #   (t, φt, dφt, H) = SecA_computation_ls(t, tprec, φtprec, dφtprec, d, φtestTR, dφtestTR)
            # end

            (t, φt, dφt, H) = step_computation_ls(direction, h, t, tprec, φtestTR, dφtestTR, d, φtprec, dφtprec)

            if symmetrical        # We adjust our interval if we want it symmetrical
                                  # or not
              if ratio > stp_ls.eps2     # Very Successful iteration => augment size of
                                  # trust region
                Δp = stp_ls.aug * Δp
                Δn = stp_ls.aug * Δn
              end
            else
              if ratio > stp_ls.eps2     # Very Successful iteration=> augment size of
                                  # trust region
                Δp = stp_ls.aug * Δp
                Δn = min(-t, stp_ls.aug * Δn)
              else
                Δn = min(-t, Δn)
              end
            end

            if admissible && add_step                        # We can do an extra step if
                                                             # desired
              t_original = copy(t)
              dht = dφt + τ₀ * g₀
              ddht = H

              dN = -dφt/H
              d = TR_ls_step_computation(H, dφt, dN, Δn, Δp)

              tprec = copy(t)
              t = t + d
              ht = obj(h, t)
              dht = grad!(h, t, g)
              verboseLS && (φt = ht - h₀ - τ₀ * t * g₀)
              verboseLS && (dφt = dht - τ₀ * g₀)
              verboseLS && (ddφt = hess(h,t))
            end

            debug && PyPlot.figure(1)
            if admissible
              debug && PyPlot.scatter([t], [ht])
            else
              debug && PyPlot.scatter([t], [φt + h₀ + τ₀ * t * g₀])
            end
            iter+=1

            verboseLS && @printf("%4d %9.2e %9.2e  %9.2e %9.2e %9.2e %9.2e %9e %9.2e  %9e\n", iter,t,φt,dφt,ddφt,Δn,Δp,1,dN, ratio);
        end;
        admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)
    end;

    # recover h
    if !add_step
      ht = φt + h₀ + τ₀ * t * g₀
    end

    t > 0.0 || (verboseLS && @show t dφt )
    @assert (t > 0.0) && (!isnan(t)) "invalid step"

    return (t, t_original, true, ht, iter, 0, tired)

end
