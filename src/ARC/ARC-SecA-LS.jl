export ARC_SecA_ls
function ARC_SecA_ls(h :: AbstractLineFunction,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   eps1 = 0.1,
                   eps2 = 0.7,
                   red = 0.15,
                   aug = 10,
                   α=1.0,
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

    iter = 0
    seck = 1.0 #(gt-g₀)
    t=0.0

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    φt = φ(t)

    dφt = dφ(t)

    q(d)=φt + dφt*d + 0.5*seck*d^2

    # test d'arrêt sur dφ
    ɛa = (τ₁-τ₀)*g₀
    ɛb = -(τ₁+τ₀)*g₀

    verbose && println("\n ɛa ",ɛa," ɛb ",ɛb," h(0) ", h₀," h₀' ",g₀)
    admissible = false
    tired=iter>maxiter
    verbose && @printf("   iter   t       φt        dφt        α\n");
    verbose && @printf(" %4d %9.2e %9.2e  %9.2e %9.2e\n", iter,t,φt,dφt,α);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations

        #step computation
        discr=seck^2-4*(dφt/α)
        if discr<0
          discr=seck^2+4*(dφt/α)
        end

        if seck<0
          dNp=(-seck+sqrt(discr))/(2/α)
        else
          dNp=-2*dφt/(seck+sqrt(discr))
        end         

        dNn=(-seck-sqrt(discr))/(2/α)

        if q(dNp)<q(dNn)
          d=dNp
        else
          d=dNn
        end

        φtestTR = φ(t+d)
        dφtestTR= dφ(t+d)
        # test d'arrêt sur dφ

        pred = dφt*d + 0.5*seck*d^2
        if pred >-1e-10
          ared=(dφt+dφtestTR)*d/2
        else
          ared=φtestTR-φt
        end

        tprec = t;
        φtprec=φt;
        dφtprec = dφt;
        ratio = ared / pred

        if ratio < eps1  # Unsuccessful
            α=red*α
            verbose && @printf("U %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e  \n", iter,t,φt,α,t+d,φtestTR);
        else             # Successful
          t = t + d

          dφt = dφtestTR
          φt = φtestTR

          s = t-tprec
          y = dφt - dφtprec

          Γ=3*(dφt+dφtprec)*s-6*(φt-φtprec)
          if (y*s+Γ) < eps(Float64)*s^2
            yt=y
          else
            yt=y+Γ/s
          end

          seck=yt/s
            if ratio > eps2
            α=aug*α
            end
            admissible = (dφt>=ɛa) & (dφt<=ɛb)  # Wolfe, Armijo garanti par la
                                                # descente
            verbose && @printf("S %4d %9.2e %9.2e  %9.2e  %9.2e \n", iter,t,φt,dφt,α);
        end;


        iter=iter+1
        tired=iter>maxiter
    end;

    # recover h
    ht = φt + h₀ + τ₀*t*g₀

    return (t,true, ht,iter,0)  #pourquoi le true et le 0?

end
