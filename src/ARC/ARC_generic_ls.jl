export ARC_generic_ls
function ARC_generic_ls(h :: LineModel,
                        h₀ :: Float64,
                        g₀ :: Float64,
                        g :: Array{Float64,1};
                        eps1 :: Float64 = 0.25,
                        eps2 :: Float64 = 0.75,
                        red :: Float64 = 0.5,
                        aug :: Float64 = 5.0,
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
                        weak_wolfe :: Bool = false,
                        kwargs...)

    (τ₀ == 1.0e-4) || (check_param && warn("Different parameters"))

    if check_slope
      (abs(g₀ - grad(h, 0.0)) < 1e-4) || warn("wrong slope")
      verboseLS && @show h₀ obj(h, 0.0) g₀ grad(h,0.0)
    end

    (t,ht,gt,A,W,ɛa,ɛb)=init_ARC(h,h₀,g₀,g,τ₀,τ₁)
    if weak_wolfe
      ɛb = Inf
    end

    if A && W
      return (t,copy(t), true, ht, 0, 0, false)
    end

    # Specialized TR for handling non-negativity constraint on t
    # Trust region parameters

    iter = 0

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    if direction=="Nwt"
      ddφ(t) = hess(h,t)
    end

    φt = φ(t)

    dφt = dφ(t)

    ddφt = NaN

    if direction=="Nwt"
      ddφt = ddφ(t)
      dersec = copy(ddφt)
      #q(d) = φt + dφt*d + 0.5*ddφt*d^2
    elseif direction=="Sec" || direction=="SecA"
      seck=1.0
      dersec = copy(seck)
      #q(d)=φt + dφt*d + 0.5*seck*d^2
    end

    if A   #version 3
      Δ = 100.0/abs(φt)
    else
      if t == 1.0
        seck = Sec_computation_ls(t, 0.0, (1.0 - τ₀) * g₀, 1.0, φt,dφt)[4]
        Δ = 1.0/abs(seck)
      else
        Δ = 1.0/abs(1000.0)
      end
    end
    verboseLS && println("   ϵₐ = $ɛa ϵᵦ = $ɛb")


    admissible = false
    tired=iter>maxiterLS
    t_original = NaN

    debug && PyPlot.figure(1)
    debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])

    verboseLS && @printf("   iter   t       φt        dφt        ddφt        Δ        Successful        dN\n");
    verboseLS && @printf("%4d %9.2e  %9.2e  %9.2e %9.2e  %9.2e\n", iter,t,φt,dφt,ddφt,Δ);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations

        if direction=="Nwt"
          d=ARC_step_computation(ddφt,dφt,Δ; kwargs...)
        elseif direction=="Sec" || direction=="SecA"
          d=ARC_step_computation(seck,dφt,Δ; kwargs...)
        end

        φtestTR = φ(t+d)
        dφtestTR= dφ(t+d)

        verboseLS && @show t, d, φtestTR, dφtestTR

        # test d'arrêt sur dφ
        if direction=="Nwt"
          (pred,ared,ratio)=pred_ared_computation(dφt,φt,ddφt,d,φtestTR,dφtestTR)
        elseif direction=="Sec" || direction=="SecA"
          (pred,ared,ratio)=pred_ared_computation(dφt,φt,seck,d,φtestTR,dφtestTR)
        end

        if direction=="Nwt"
          tprec = t
          φtprec = φt
          dφtprec = dφt
          ddφtprec = ddφt
        elseif direction=="Sec"
          tprec = t
          dφtprec = dφt
        elseif direction=="SecA"
          tprec = t
          φtprec=φt
          dφtprec = dφt
        end

        if (t + d < 0.0) && (ratio < eps1)  # Unsuccessful
          Δ=red*Δ
          iter += 1
          verboseLS && @printf("%4d %9.2e %9.2e  %9.2e  %9.2e  %9.2e %9.2e %7.2e\n", iter,t,φt,dφt,ddφt,Δ,0,d);
        else             # Successful

          if direction=="Nwt"
            (t,φt,dφt,ddφt) = Nwt_computation_ls(t, d , φtestTR, dφtestTR, h)
          elseif direction=="Sec"
            verboseLS && @show (t, tprec, dφtprec, d, φtestTR,dφtestTR)
            (t,φt,dφt,seck) = Sec_computation_ls(t, tprec, dφtprec, d, φtestTR,dφtestTR)
            verboseLS && @show (t,φt,dφt,seck)
          elseif direction=="SecA"
            (t,φt,dφt,seck) = SecA_computation_ls(t, tprec, φtprec, dφtprec, d, φtestTR,dφtestTR)
          end

          if ratio > eps2
            Δ = aug*Δ
          end

          admissible = ((dφt>=ɛa) & (dφt<=ɛb)) # Wolfe, Armijo garanti par la
                                               # descente

          if admissible && add_step && (n_add_step < 1)
            n_add_step +=1
            admissible = false
          end

          debug && PyPlot.figure(1)
          debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])
          iter += 1
          tired = iter > maxiterLS

          verboseLS && @show t d
          verboseLS && @printf("%4d %9.2e %9.2e  %9.2e  %9.2e  %9.2e  %9.2e %7.2e \n", iter,t,φt,dφt,ddφt,Δ,1,d);
        end;
    end;

    # recover h
    ht = φt + h₀ + τ₀*t*g₀

    @assert (t > 0.0) && (!isnan(t)) "invalid step"

    return (t, t_original, true, ht, iter, 0, tired)  #pourquoi le true et le 0?

end
