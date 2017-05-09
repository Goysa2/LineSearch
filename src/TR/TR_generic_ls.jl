export TR_generic_ls
function TR_generic_ls(h :: AbstractLineFunction,
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
                       verbose :: Bool=false,
                       direction :: String="Nwt",
                       kwargs...)

    #println("on est dans TR_generic_ls direction:",direction )

    t = 1.0
    (t,ht,gt,A_W,Δp,Δn,ɛa,ɛb)=init_TR(h,h₀,g₀,g,τ₀,τ₁)

    if A_W
      return (t,true,ht,0.0,0.0)
    end

    iter = 0

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    # le reste de l'algo minimise la fonction φ...
    # par conséquent, le critère d'Armijo sera vérifié φ(t)<φ(0)=0
    φt = φ(t)          # on sait que φ(0)=0

    dφt = dφ(t) # connu dφ(0)=(1.0-τ₀)*g₀

    if direction=="Nwt"
      ddφt = hess(h,t)
    elseif direction=="Sec" || direction=="SecA"
      seck=1.0
      #println("on a seck")
    end

    if direction=="Nwt"
      q(d) = φt + dφt*d + 0.5*ddφt*d^2
    elseif direction=="Sec" || direction=="SecA"
      q(d)=φt + dφt*d + 0.5*seck*d^2
      #println("on a q(d)")
    end

    admissible = false
    tired=iter > maxiter
    verbose && @printf("   iter   t       φt        dφt        Δn        Δp        t+d        φtestTR\n")
    verbose && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e \n", iter,t,φt,dφt,Δn,Δp);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations

        if direction=="Nwt"
          dN = -dφt/ddφt; # point stationnaire de q(d)
          d=TR_ls_step_computation(ddφt,dφt,dN,Δn,Δp)
        elseif direction=="Sec" || direction=="SecA"
          dN = -dφt/seck
          d=TR_ls_step_computation(seck,dφt,dN,Δn,Δp)
          #println("on a la direction d")
        end

        φtestTR = φ(t+d)
        dφtestTR= dφ(t+d)
        # test d'arrêt sur dφ

        if direction=="Sec"
          tprec = t
          #φtprec = φt
          dφtprec = dφt
          # ddφtprec = ddφt
          #println("on a tprec et dφtprec")
        elseif direction=="SecA"
          tprec = t
          φtprec = φt
          dφtprec = dφt
        end


        if direction=="Nwt"
          (pred,ared,ratio)=pred_ared_computation(dφt,φt,ddφt,d,φtestTR,dφtestTR)
        elseif direction=="Sec"
          (pred,ared,ratio)=pred_ared_computation(dφt,φt,seck,d,φtestTR,dφtestTR)
        end

        if ratio < eps1  # Unsuccessful
            Δp = red*Δp
            Δn = red*Δn
            verbose && @printf("U %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e  %9.2e %9.2e\n", iter,t,φt,dφt,Δn,Δp,t+d,φtestTR);
        else             # Successful
            if direction=="Nwt"
              (t,φt,dφt,ddφt)=Nwt_computation_ls(t,d,φtestTR,h,dφ)
            elseif direction=="Sec"
              (t,φt,dφt,s,y,seck)=Sec_computation_ls(t,tprec, dφtprec, dφ, d, φtestTR)
              #println("on a les paramètres de la sécante")
            elseif direction=="SecA"
              (t,φt,dφt,s,y,seck)=SecA_computation_ls(t, tprec, φtprec, dφtprec,dφ, d, φtestTR)
            end


            if ratio > eps2
                Δp = aug * Δp
                Δn = min(-t, aug * Δn)
            else
                Δn = min(-t, Δn)
            end

            admissible = (dφt>=ɛa) & (dφt<=ɛb)  # Wolfe, Armijo garanti par la
                                                # descente
            verbose && @printf("N %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e \n", iter,t,φt,dφt,Δn,Δp);
        end;
        iter+=1
        tired=iter>maxiter
    end;

    # recover h
    ht = φt + h₀ + τ₀*t*g₀

    return (t,true, ht, iter,0)  #pourquoi le true et le 0?

end
