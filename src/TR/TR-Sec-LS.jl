export TR_Sec_ls
function TR_Sec_ls(h :: AbstractLineFunction,
                  h₀ :: Float64,
                  g₀ :: Float64,
                  g :: Array{Float64,1};
                  τ₀ :: Float64=1.0e-4,
                  τ₁ :: Float64=0.9999,
                  maxiter :: Int=50,
                  verbose :: Bool=false)

    t = 1.0
    ht = obj(h,t)
    gt = grad!(h, t, g)

    cond_Armijo=Armijo(t,ht,gt,h₀,g₀,τ₀)
    cond_Wolfe=Wolfe(gt,g₀,τ₁)

    if cond_Armijo && cond_Wolfe
      return (t, true, ht, 0,0)
    end

    if cond_Armijo
      Δp = 1.0  # >=0
      Δn = -1.0  # <=0
      t=1.0
    else
      Δp = 1.0  # >=0
      Δn = 0.0  # <=0
      t=0.0
    end



    # Specialized TR for handling non-negativity constraint on t
    # Trust region parameters
    eps1 = 0.1
    eps2 = 0.7
    red = 0.15
    aug = 10
    #Δp = 1.0  # >=0
    #Δn = 0.0  # <=0

    iter = 0
    seck = 1.0 #(gt-g₀)
    #t=0.0



    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    # le reste de l'algo minimise la fonction φ...
    # par conséquent, le critère d'Armijo sera vérifié φ(t)<φ(0)=0
    φt = φ(t)          # on sait que φ(0)=0

    dφt = dφ(t) # connu dφ(0)=(1.0-τ₀)*g₀
    # Version avec la sécante: modèle quadratique
    q(d)=φt + dφt*d + 0.5*seck*d^2

    # test d'arrêt sur dφ
    ɛa = (τ₁-τ₀)*g₀
    ɛb = -(τ₁+τ₀)*g₀

    admissible = false
    tired=iter > maxiter
    verbose && @printf("   iter   t       φt        dφt        Δn        Δp        t+d        φtestTR\n");
    verbose && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e\n", iter,t,φt,dφt,Δn,Δp);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations
      if iter==0
        #println("on rentre la dedans")
      end
        dS = -dφt/seck; # point stationnaire de q(d)

        if (q(Δp)<q(Δn)) | (Δn==0.0)
            d=Δp
        else
            d=Δn
        end

        if  (dS < Δp) & (dS > Δn) & (q(d)>q(dS))
            d=dS
        end

        φtestTR = φ(t+d)
        dφtestTR= dφ(t+d)


        pred = dφt*d + 0.5*seck*d^2
        if pred >-1e-10
          ared=(dφt+dφtestTR)*d/2
        else
          ared=φtestTR-φt
        end

        tprec = t;
        dφtprec = dφt;
        ratio = ared / pred

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
