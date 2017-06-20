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
                       maxiter :: Int64=50,
                       verboseLS :: Bool=false,
                       direction :: String="Nwt",
                       check_param :: Bool = false,
                       debug :: Bool = true,
                       kwargs...)

    (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))

    t = 1.0
    (t,ht,gt,A_W,Δp,Δn,ɛa,ɛb)=init_TR(h,h₀,g₀,g,τ₀,τ₁;kwargs...)

    if A_W
      return (t,true,ht,0.0,0.0,false, h.f_eval, h.g_eval, h.h_eval)
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
    tired=iter > maxiter

    debug && PyPlot.figure(1)
    debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])

    verboseLS && @printf("   iter   t       φt        dφt        ddφt        Δn        Δp        Successful        dN\n")
    verboseLS && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e  %9.2e %9.2e \n", iter,t,φt,dφt,ddφt,Δn,Δp);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations

        if direction=="Nwt"
          dN = -dφt/ddφt; # point stationnaire de q(d)
          d=TR_ls_step_computation(ddφt,dφt,dN,Δn,Δp)
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
            verboseLS && @printf("%4d %9.2e %9.2e  %9.2e %9.2e %9.2e %9.2e  %9.2e  %9.2e\n", iter,t,φt,dφt,ddφt,Δn,Δp,0,dN);
        else             # Successful
            if direction=="Nwt"
              (t,φt,dφt,ddφt)=Nwt_computation_ls(t,d,φtestTR,dφtestTR,h)
            elseif direction=="Sec"
              (t,φt,dφt,s,y,seck)=Sec_computation_ls(t,tprec, dφtprec, d, φtestTR,dφtestTR)
            elseif direction=="SecA"
              (t,φt,dφt,s,y,seck)=SecA_computation_ls(t, tprec, φtprec, dφtprec, d, φtestTR,dφtestTR)
            end


            if ratio > eps2
                Δp = aug * Δp
                Δn = min(-t, aug * Δn)
            else
                Δn = min(-t, Δn)
            end

            admissible = (dφt>=ɛa) & (dφt<=ɛb)  # Wolfe, Armijo garanti par la
                                                # descente

            debug && PyPlot.figure(1)
            debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])
            iter+=1
            verboseLS && @printf("%4d %9.2e %9.2e  %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter,t,φt,dφt,ddφt,Δn,Δp,1,dN);
        end;
        tired=iter>maxiter
    end;

    # println("tired=",tired)

    # recover h
    ht = φt + h₀ + τ₀*t*g₀
    # println("(t,true, ht, iter, 0, tired)=",(t,true, ht, iter, 0, tired))

    return (t,true, ht, iter, 0, tired, h.f_eval, h.g_eval, h.h_eval)  #pourquoi le true et le 0?

end
