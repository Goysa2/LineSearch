export TR_Sec_ls
function TR_Sec_ls(h :: AbstractLineFunction,
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
                   verbose :: Bool=false)

    (t,ht,gt,A_W,Δp,Δn,ɛa,ɛb)=init_TR(h,h₀,g₀,g,τ₀,τ₁)

    if A_W
      return (t,true,ht,0.0,0.0)
    end

    iter = 0
    seck = 1.0 #(gt-g₀)

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    # le reste de l'algo minimise la fonction φ...
    # par conséquent, le critère d'Armijo sera vérifié φ(t)<φ(0)=0
    φt = φ(t)          # on sait que φ(0)=0

    dφt = dφ(t) # connu dφ(0)=(1.0-τ₀)*g₀
    # Version avec la sécante: modèle quadratique
    q(d)=φt + dφt*d + 0.5*seck*d^2

    admissible = false
    tired=iter > maxiter
    verbose && @printf("   iter   t       φt        dφt        Δn        Δp        t+d        φtestTR\n");
    verbose && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e\n", iter,t,φt,dφt,Δn,Δp);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations
      if iter==0
        #println("on rentre la dedans")
      end
        dS = -dφt/seck; # point stationnaire de q(d)

        d=TR_ls_step_computation(ddφt,dφt,dS,Δn,Δp)

        φtestTR = φ(t+d)
        dφtestTR= dφ(t+d)

        (pred,ared,ratio)=pred_ared_computation(dφt,φt,ddφt,d,φtestTR,dφtestTR)

        if ratio < eps1  # Unsuccessful
            Δp = red*Δp
            Δn = red*Δn
            verbose && @printf("U %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e  %9.2e %9.2e\n", iter,t,φt,dφt,Δn,Δp,t+d,φtestTR);
        else             # Successful
            t = t + d
            dφt = dφ(t)
            φt = φtestTR
            s = t-tprec
            y = dφt - dφtprec
            seck = y/s
            if ratio > eps2
                Δp = aug * Δp
                Δn = min(-t, aug * Δn)
            else
                Δn = min(-t, Δn)
            end
            admissible = (dφt>=ɛa) & (dφt<=ɛb)  # Wolfe, Armijo garanti par la
                                                # descente
            verbose && @printf("S %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e \n", iter,t,φt,dφt,Δn,Δp);
        end;
        iter+=1
        tired=iter>maxiter
    end;

    ht = φt + h₀ + τ₀*t*g₀
    return (t,true, ht, iter,0)  #pourquoi le true et le 0?
end
