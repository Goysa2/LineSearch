export TR_Nwt_ls
function TR_Nwt_ls(h :: AbstractLineFunction,
                  h₀ :: Float64,
                  g₀ :: Float64,
                  g :: Array{Float64,1};
                  τ₀ :: Float64=1.0e-4,
                  τ₁ :: Float64=0.9999,
                  maxiter :: Int=50,
                  verbose :: Bool=false)

    t = 1.0
    (t,ht,gt,A_W,Δp,Δn,eps1,eps2,red,aug,ɛa,ɛb)=init_TR(h,h₀,g₀,g,τ₀,τ₁)

    if A_W
      return (t,true,ht,0.0,0.0)
    end

    kt=hess(h,t)

    iter = 0
    seck = 1.0 #(gt-g₀)

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    # le reste de l'algo minimise la fonction φ...
    # par conséquent, le critère d'Armijo sera vérifié φ(t)<φ(0)=0
    φt = φ(t)          # on sait que φ(0)=0

    dφt = dφ(t) # connu dφ(0)=(1.0-τ₀)*g₀

    ddφt = hess(h,t)
    # Version avec la sécante: modèle quadratique
    q(d) = φt + dφt*d + 0.5*ddφt*d^2

    admissible = false
    tired=iter > maxiter
    verbose && @printf("   iter   t       φt        dφt        Δn        Δp        t+d        φtestTR\n")
    verbose && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e \n", iter,t,φt,dφt,Δn,Δp);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations

        dN = -dφt/ddφt; # point stationnaire de q(d)

        if (q(Δp)<q(Δn)) | (Δn==0.0)
            d=Δp
        else
            d=Δn
        end

        if  (dN < Δp) & (dN > Δn) & (q(d)>q(dN))
            d=dN
        end

        φtestTR = φ(t+d)
        dφtestTR= dφ(t+d)
        # test d'arrêt sur dφ

        pred = dφt*d + 0.5*kt*d^2
        #assert(pred<0)   # How to recover? As is, it seems to work...
        if pred >-1e-10
          ared=(dφt+dφtestTR)*d/2
        else
          ared=φtestTR-φt
        end

        # tprec = t
        # φtprec = φt
        # dφtprec = dφt
        # ddφtprec = ddφt

        ratio = ared / pred; # inclure ici le truc numérique de H & Z

        if ratio < eps1  # Unsuccessful
            Δp = red*Δp
            Δn = red*Δn
            verbose && @printf("U %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e  %9.2e %9.2e\n", iter,t,φt,dφt,Δn,Δp,t+d,φtestTR);
        else             # Successful
            t = t + d

            φt = φtestTR
            dφt = dφ(t)
            ddφt = hess(h,t)

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
