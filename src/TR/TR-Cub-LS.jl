export TR_Cub_ls
function TR_Cub_ls(h :: AbstractLineFunction,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   τ₀ :: Float64=1.0e-4,
                   τ₁ :: Float64=0.9999,
                   maxiter :: Int64=50,
                   verbose :: Bool=false)

    (t,ht,gt,A_W,Δp,Δn,eps1,eps2,red,aug,ɛa,ɛb)=init_TR(h,h₀,g₀,g,τ₀,τ₁)

    if A_W
      return (t,true,ht,0.0,0.0)
    end

    iter = 0
    gt=grad(h,t)
    dN=-gt #pour la premiere iteration

    A=0.5
    B=0.0
    
    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    # le reste de l'algo minimise la fonction φ...
    # par conséquent, le critère d'Armijo sera vérifié φ(t)<φ(0)=0
    φt = φ(t)          # on sait que φ(0)=0

    dφt = dφ(t) # connu dφ(0)=(1.0-τ₀)*g₀
    # Version avec la sécante: modèle quadratique
    q(d)=φt + dφt*d + A*d^2 + B*d^3

    verbose &&println("\n ɛa ",ɛa," ɛb ",ɛb," h(0) ", h₀," h₀' ",g₀)
    admissible = false
    tired=iter > maxiter
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
        end;
        iter+=1
        tired=iter>maxiter
    end

    # recover h
    ht = φt + h₀ + τ₀*t*g₀
    return (t,true, ht, iter,0)   #pourquoi le true et le 0?

end
