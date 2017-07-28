export ARC_generic_ls
function ARC_generic_ls(h :: LineModel,
                        h₀ :: Float64,
                        g₀ :: Float64,
                        g :: Array{Float64,1};
                        stp_ls :: TStopping_LS = TStopping_LS(),
                        Δ :: Float64 = 1.0,
                        τ₀ :: Float64=1.0e-4,
                        τ₁ :: Float64=0.9999,
                        maxiterLS :: Int64=50,
                        verboseLS :: Bool=false,
                        direction :: String="Nwt",
                        check_param :: Bool = false,
                        check_slope :: Bool = false,
                        add_step :: Bool = true,
                        n_add_step :: Int64 = 0,
                        debug :: Bool = false,
                        kwargs...)

    (τ₀ == 1.0e-4) || (check_param && warn("Different parameters"))

    if check_slope
      (abs(g₀ - grad(h, 0.0)) < 1e-4) || warn("wrong slope")
      verboseLS && @show h₀ obj(h, 0.0) g₀ grad(h,0.0)
    end

    #println("on rentre dans ARC τ₀ = $τ₀ τ₁ = $τ₁")

    (t, ht, gt, A, W) = init_ARC(h, h₀, g₀, g, τ₀, τ₁)
    start_ls!(g, stp_ls, τ₀, τ₁, h₀, g₀; eps1 = 0.25, eps2 = 0.75, kwargs...)

    if A && W
      return (t, copy(t), true, ht, 0, 0, false)
    end

    # Specialized TR for handling non-negativity constraint on t
    # Trust region parameters

    iter = 0

    φ(t) = obj(h, t) - h₀ - τ₀ * t * g₀  # fonction et
    dφ(t) = grad!(h, t, g) - τ₀ * g₀      # dérivée

    #φt = φ(t)          # on sait que φ(0)=0
    #dφt = dφ(t)        # connu dφ(0)=(1.0-τ₀)*g₀
    φt = ht - h₀ - τ₀*t*g₀
    dφt = gt - τ₀*g₀
    if t == 0.0
        φt = 0.0
        dφt = (1 - τ₀) * g₀
    end

    ddφt = NaN

    if direction == "Nwt"
      H = hess(h, t)
    elseif direction == "Sec" || direction == "SecA"
      H = 1.0
    end

    if A   #version 3
      Δ = 100.0 / abs(φt)
    else
      Δ = 1.0 / abs(H)
    end

    verboseLS && println("ɛa = $(stp_ls.ɛa) ɛb = $(stp_ls.ɛb)")

    admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)
    t_original = NaN

    debug && PyPlot.figure(1)
    debug && PyPlot.scatter([t],[φt + h₀ + τ₀ * t * g₀])

    verboseLS && @printf("   iter   t       φt        dφt        ddφt        Δ")
    verboseLS && @printf("   Successful        dN        ratio\n")
    verboseLS && @printf("%4d %9.2e  %9.2e  %9.2e %9.2e  %9.2e\n",
                         iter, t, φt, dφt, ddφt, Δ);

    while !(admissible | tired) # admissible: respecte armijo et wolfe,
                                # tired: nb d'itérations

        d = ARC_step_computation(H, dφt, Δ; kwargs...)

        φtestTR = φ(t + d)
        dφtestTR = dφ(t + d)

        # test d'arrêt sur dφ
        (pred, ared, ratio) =
               pred_ared_computation(dφt, φt, H, d, φtestTR, dφtestTR)

        if direction == "Nwt"
          tprec = t
          φtprec = φt
          dφtprec = dφt
          ddφtprec = ddφt
        elseif direction == "Sec"
          tprec = t
          dφtprec = dφt
        elseif direction == "SecA"
          tprec = t
          φtprec=φt
          dφtprec = dφt
        end

        if (t + d < 0.0) || (ratio < stp_ls.eps1)  # Unsuccessful
          Δ = stp_ls.red * Δ
          iter += 1
          verboseLS &&
            @printf("%4d %9.2e %9.2e  %9.2e  %9.2e  %9.2e %9.2e %7.2e\n",
                    iter, t, φt, dφt, ddφt, Δ, 0, d);
        else             # Successful

          if direction == "Nwt"
            (t, φt, dφt , H) = Nwt_computation_ls(t, d , φtestTR, dφtestTR, h)
          elseif direction == "Sec"
            (t, φt, dφt, H) = Sec_computation_ls(t, tprec, dφtprec, d,
                                                 φtestTR, dφtestTR)
          elseif direction == "SecA"
            (t, φt, dφt, H ) =SecA_computation_ls(t, tprec, φtprec, dφtprec, d,
                                                  φtestTR, dφtestTR)
          end

          if ratio > stp_ls.eps2
            Δ = stp_ls.aug*Δ
          end

          if admissible && add_step && (n_add_step < 1)
            n_add_step += 1
            admissible = false
          end

          debug && PyPlot.figure(1)
          debug && PyPlot.scatter([t],[φt + h₀ + τ₀ * t * g₀])
          iter += 1

          verboseLS &&
            @printf("%4d %9.2e %9.2e  %9.2e  %9.2e  %9.2e  %9.2e %7.2e \n",
                    iter, t, φt, dφt, ddφt, Δ, 1, d);
        end;
        admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)
    end;

    # recover h
    ht = φt + h₀ + τ₀ * t * g₀

    @assert (t > 0.0) && (!isnan(t)) "invalid step"

    return (t, t_original, true, ht, iter, 0, tired)

end
