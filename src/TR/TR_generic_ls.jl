export TR_generic_ls
function TR_generic_ls(h :: AbstractLineFunction2,
                       h₀ :: Float64,
                       g₀ :: Float64,
                       g :: Array{Float64,1};
                       τ₀ :: Float64=1.0e-4,
                       eps1 :: Float64 = 0.1,
                       eps2 :: Float64 = 0.7,
                       red :: Float64 = 0.15,
                       aug :: Float64= 10.0,
                       τ₁ :: Float64=0.9999,
                       maxiterLS :: Int64=50,
                       verboseLS :: Bool=false,
                       direction :: String="Nwt",
                       check_param :: Bool = false,
                       debug :: Bool = false,
                       kwargs...)

    (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))

    t = 1.0
    t_original = NaN
    (t,ht,gt,A_W,Δp,Δn,ɛa,ɛb)=init_TR(h,h₀,g₀,g,τ₀,τ₁;kwargs...)

    if A_W
      return (t,t_original,true,ht,0.0,0.0,false, h.f_eval, h.g_eval, h.h_eval)
    end

    iter = 0

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    # le reste de l'algo minimise la fonction φ...
    # par conséquent, le critère d'Armijo sera vérifié φ(t)<φ(0)=0
    φt = φ(t)          # on sait que φ(0)=0

    dφt = dφ(t) # connu dφ(0)=(1.0-τ₀)*g₀

    ddφt = NaN

    if direction=="Nwt"
      ddφt = hess(h,t)
    elseif direction=="Sec" || direction=="SecA"
      seck=1.0
    end

    if direction=="Nwt"
      q(d) = φt + dφt*d + 0.5*ddφt*d^2
    elseif direction=="Sec" || direction=="SecA"
      q(d)=φt + dφt*d + 0.5*seck*d^2
    end

    verboseLS && println("   ϵₐ = $ɛa ϵᵦ = $ɛb")

    admissible = false
    tired=iter > maxiterLS

    debug && PyPlot.figure(1)
    debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])

    verboseLS && @printf("  iter   t       φt        dφt        ddφt        Δn        Δp        Successful        dN\n")
    verboseLS && @printf("%4d %9.2e  %9.2e %9.2e %9.2e %9.2e %9.2e \n", iter,t,φt,dφt,ddφt,Δn,Δp);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations
        if direction=="Nwt"
          dN = -dφt/ddφt; # point stationnaire de q(d)
          # println("avant le calcul de d on a dN = $dN ddφt = $ddφt dφt = $dφt  ")
          d=TR_ls_step_computation(ddφt,dφt,dN,Δn,Δp)
          # println("apres le calcul d = $d")
        elseif direction=="Sec" || direction=="SecA"
          dN = -dφt/seck
          d=TR_ls_step_computation(seck,dφt,dN,Δn,Δp)
        end

        φtestTR = φ(t+d)
        dφtestTR= dφ(t+d)
        # test d'arrêt sur dφ

        if direction=="Sec"
          tprec = t
          dφtprec = dφt
        elseif direction=="SecA"
          tprec = t
          φtprec = φt
          dφtprec = dφt
        end


        if direction=="Nwt"
          (pred,ared,ratio)=pred_ared_computation(dφt,φt,ddφt,d,φtestTR,dφtestTR)
        elseif direction=="Sec" || direction=="SecA"
          (pred,ared,ratio)=pred_ared_computation(dφt,φt,seck,d,φtestTR,dφtestTR)
        end

        if ratio < eps1  # Unsuccessful
            Δp = red*Δp
            Δn = red*Δn
            iter+=1
            verboseLS && @printf("%4d %9.2e %9.2e  %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter,t,φt,dφt,ddφt,Δn,Δp,0,dN);
        else             # Successful
            if direction=="Nwt"
              # println("avant de recalculer t= $t d = $d ")
              (t,φt,dφt,ddφt)=Nwt_computation_ls(t,d,φtestTR,dφtestTR,h)
              # println("après recalcul t = $t")
            elseif direction=="Sec"
              (t,φt,dφt,seck)=Sec_computation_ls(t,tprec, dφtprec, d, φtestTR,dφtestTR)
            elseif direction=="SecA"
              (t,φt,dφt,seck)=SecA_computation_ls(t, tprec, φtprec, dφtprec, d, φtestTR,dφtestTR)
            end


            if ratio > eps2
                Δp = aug * Δp
                Δn = min(-t, aug * Δn)
            else
                Δn = min(-t, Δn)
            end

            admissible = (dφt>=ɛa) & (dφt<=ɛb)    # Wolfe, Armijo garanti par la
                                                  # descente

            # if admissible
            #   t_original = copy(t)
            #   dht = dφt + τ₀ * g₀
            #   ddht = ddφt
            #   if direction=="Nwt"
            #     dN = -dφt/ddφt; # point stationnaire de q(d)
            #     d=TR_ls_step_computation(ddφt,dφt,dN,Δn,Δp)
            #   elseif direction=="Sec" || direction=="SecA"
            #     dN = -dφt/seck
            #     d=TR_ls_step_computation(seck,dφt,dN,Δn,Δp)
            #   end
            #   tprec= copy(t)
            #   t = t + d
            #   ht = obj(h,t)
            #   dht = grad!(h,t,g)
            #   verboseLS && (φt = ht - h₀ - τ₀ * t * g₀)
            #   verboseLS && (dφt = dht - τ₀ * g₀)
            #   verboseLS && (ddφt = hess(h,t))
            # end

            debug && PyPlot.figure(1)
            if admissible
              debug && PyPlot.scatter([t],[ht])
            else
              debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])
            end
            iter+=1
            verboseLS && @printf("%4d %9.2e %9.2e  %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter,t,φt,dφt,ddφt,Δn,Δp,1,dN);
        end;
        tired=iter>maxiterLS
    end;

  #  println(" on est sortie")

  #  println("t_original = $t_original")

    # recover h
    # ht = φt #+ h₀ + τ₀*t*g₀
    # dht = dφt + τ₀*g₀

    return (t, t_original, true, ht, iter, 0, tired, h.f_eval, h.g_eval, h.h_eval)  #pourquoi le true et le 0?

end
