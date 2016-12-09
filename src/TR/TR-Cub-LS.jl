export TR_Cub_ls
function TR_Cub_ls(h :: AbstractLineFunction,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   τ₀ :: Float64=1.0e-4,
                   τ₁ :: Float64=0.9999,
                   max_eval :: Int64=100,
                   verbose :: Bool=false)

    t = 1.0
    ht = obj(h,t)
    gt = grad!(h, t, g)
    if Armijo(t,ht,gt,h₀,g₀,τ₀) && Wolfe(gt,g₀,τ₁)
        nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
        return (t, true, ht,nftot)
    end

    # Specialized TR for handling non-negativity constraint on t
    # Trust region parameters
    eps1 = 0.1
    eps2 = 0.7
    red = 0.15
    aug = 10
    Δp = 1.0  # >=0
    Δn = 0.0  # <=0

    iter = 0
    # seck = 1.0 #(gt-g₀)
    # dN=-gt #pour la première itération
    t=0.0
    gt=grad(h,t)
    dN=-gt #pour la premiere iteration

    A=0.5
    B=0.0


    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    # le reste de l'algo minimise la fonction φ...
    # par conséquent, le critère d'Armijo sera vérifié φ(t)<φ(0)=0
    φt = 0.0          # on sait que φ(0)=0

    dφt = (1.0-τ₀)*g₀ # connu dφ(0)=(1.0-τ₀)*g₀
    # Version avec la sécante: modèle quadratique
    q(d)=φt + dφt*d + A*d^2 + B*d^3

    # test d'arrêt sur dφ
    ɛa = (τ₁-τ₀)*g₀
    ɛb = -(τ₁+τ₀)*g₀

    verbose &&println("\n ɛa ",ɛa," ɛb ",ɛb," h(0) ", h₀," h₀' ",g₀)
    admissible = false
    nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
    tired=nftot > max_eval
    verbose && @printf("   iter   t       φt        dφt        Δn        Δp  \n");
    verbose && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e \n", iter,t,φt,dφt,Δn,Δp);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations

        if (q(Δp)<q(Δn)) | (Δn==0.0)
            d=Δp
        else
            d=Δn
        end

        cub(t)= φt + dφt*t + A*t^2 + B*t^3
        dcub(t)=dφt + 2*A*t + 3*B*t^2
        dR=roots(dcub)

        if isreal(dR)
          dR=real(dR)
          dN2=dR[1]
          if length(dR)>1
            if cub(dR[1])>cub(dR[2])
              dN2=dR[2]
            end
          end
        else
          dN2=-dφt*s/y
        end

        if (abs(dN2)<abs(Δp-Δn)) & (q(d)>q(dN2))
          d=dN2
        end

        φtestTR = φ(t+d)
        dφtestTR= dφ(t+d)
        # test d'arrêt sur dφ

        #pred = dφt*d + 0.5*seck*d^2
        pred = dφt*d + A*d^2 + B*d^3
        #assert(pred<0)   # How to recover? As is, it seems to work...
        if pred >-1e-10
          ared=(dφt+dφtestTR)*d/2
        else
          ared=φtestTR-φt
        end

        tprec = t;
        φtprec=φt
        dφtprec = dφt;
        ratio = ared / pred

        if ratio < eps1  # Unsuccessful
            Δp = red*Δp
            Δn = red*Δn
            verbose && @printf("U %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e  %9.2e %9.2e\n", iter,t,φt,dφt,Δn,Δp,t+d,φtestTR);
        else             # Successful
            t = t + d
            dφt = dφtestTR
            φt = φtestTR

            s = t-tprec
            y = dφt - dφtprec
            α=-s
            z= dφt + dφtprec + 3*(φt-φtprec)/α
            discr = z^2-dφtprec*dφt
            denom = dφt + dφtprec + 2*z
            B= 1/3*(dφt+dφtprec+2*z)/(α*α)
            A=-(dφt+z)/α

            if ratio > eps2
                Δp = aug * Δp
                Δn = min(-t, aug * Δn)
            else
                Δn = min(-t, Δn)
            end
            admissible = (dφt>=ɛa) & (dφt<=ɛb)  # Wolfe, Armijo garanti par la
                                                # descente
            verbose && @printf("S %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e \n", iter,t,φt,dφt,Δn,Δp);
            if t==tprec
              ht = φt + h₀ + τ₀*t*g₀
              nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
              return (t, true, ht,nftot)
            end
        end;


        iter=iter+1
        nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
        tired=nftot > max_eval
    end;

    # recover h
    ht = φt + h₀ + τ₀*t*g₀

    return (t, true, ht,nftot)  #pourquoi le true et le 0?

end
