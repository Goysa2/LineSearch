export ARC_Cub_ls
function ARC_Cub_ls(h :: AbstractLineFunction,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   τ₀ :: Float64=1.0e-4,
                   τ₁ :: Float64=0.9999,
                   maxiter :: Int64=50,
                   verbose :: Bool=false)

    (t,ht,gt,A_W,ɛa,ɛb)=init_TR(h,h₀,g₀,g,τ₀,τ₁)

    if A_W
      return (t,true,ht,0.0,0.0)
    end

    # Specialized TR for handling non-negativity constraint on t
    # Trust region parameters
    eps1 = 0.1
    eps2 = 0.7
    red = 0.15
    aug = 10
    α=1.0

    iter = 0
    #t=0.0
    #gt=grad(h,t)
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
    #ɛa = (τ₁-τ₀)*g₀
    #ɛb = -(τ₁+τ₀)*g₀

    verbose && println("\n ɛa ",ɛa," ɛb ",ɛb," h(0) ", h₀," h₀' ",g₀)
    admissible = false
    tired=iter>maxiter
    verbose && @printf("   iter   t       φt        dφt        α        t+d      \n");
    verbose && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e\n", iter,t,φt,dφt,α,t+d);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations

        #step computation
        Quad(t) = φt + dφt*t + A*t^2 + B*t^3 + (1/(4*α))*t^4
        dQuad(t) = dφt+2*A*t+3*B*t^2+(1/α)*t^3

        p=Poly([gₖ,2*A,3*B,(1/α)])

        dR=roots(p)

        vmin=Inf
        for i=1:length(dR)
          rr=dR[i]
          if isreal(rr)
            rr=real(rr)
            vact=Quad(rr)
            if rr*dφt<0
              if vact<vmin
                dN=rr
                vmin=vact
              end
            end
          end
        end

        d=dN

        if d==0.0
          ht = φt + h₀ + τ₀*t*g₀
          return (t,true, ht, iter+1 , 0, true)
          #L'OUTIL DE CALCUL DES RACINES DU PACKAGE Polynomials ARRONDIE À 0 CERTAINES RACINES D'OÙ LA NOUVELLE CONDITION
        end


        φtestTR = φ(t+d)
        dφtestTR= dφ(t+d)
        # test d'arrêt sur dφ

        pred = dφt*d + A*d^2 + B*d^3
        #assert(pred<0)   # How to recover? As is, it seems to work...
        if pred >-1e-10
          ared=(dφt+dφtestTR)*d/2
        else
          ared=φtestTR-φt
        end

        tprec = t;
        φtprec = φt;
        dφtprec = dφt;
        ratio = ared / pred

        if ratio < eps1  # Unsuccessful
            α=red*α
            verbose && @printf("U %4d %9.2e %9.2e  %9.2e %9.2e  %9.2e %9.2e\n", iter,t,φt,dφt,α,t+d,φtestTR);
        else             # Successful
            t = t + d

            dφt = dφtestTR
            φt = φtestTR

            s = t-tprec
            y = dφt- dφtprec

            a=-s
            z=dφt+dφtprec+3*(φt-φtprec)/a
            discr=z^2-dφt*dφtprec
            denom=dφt+dφtprec+2*z
            B=(1/3)*(dφt+dφtprec+2*z)/(a*a)
            A=-(dφt+z)/a

            if ratio > eps2
                α=aug*α
            end
            admissible = (dφt>=ɛa) & (dφt<=ɛb)  # Wolfe, Armijo garanti par la
                                                # descente
            verbose && @printf("S %4d %9.2e %9.2e  %9.2e   %9.2e \n", iter,t,φt,dφt,α);
        end;
        iter=iter+1
        tired=iter>maxiter
    end;

    # recover h
    ht = φt + h₀ + τ₀*t*g₀

    return (t,true, ht, iter,0)  #pourquoi le true et le 0?

end
