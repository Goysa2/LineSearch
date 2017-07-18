export ARC_Cub_ls
function ARC_Cub_ls(h :: LineModel,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   eps1 :: Float64 = 0.25,
                   eps2 :: Float64 = 0.75,
                   red :: Float64 = 0.5,
                   aug :: Float64 = 5.0,
                   Δ :: Float64 = 1.0,
                   τ₀ :: Float64=1.0e-4,
                   τ₁ :: Float64=0.9999,
                   maxiterLS :: Int64=50,
                   verboseLS :: Bool=false,
                   check_param :: Bool = false,
                   check_slope :: Bool = false,
                   add_step :: Bool = true,
                   n_add_step :: Int64 = 0,
                   debug :: Bool = false,
                   weak_wolfe :: Bool = false,
                   kwargs...)

    (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))

    if check_slope
      (abs(g₀ - grad(h, 0.0)) < 1e-4) || warn("wrong slope")
      verboseLS && @show h₀ obj(h, 0.0) g₀ grad(h,0.0)
    end

    (t,ht,gt,Ar,W,ɛa,ɛb)=init_ARC(h,h₀,g₀,g,τ₀,τ₁)

    if weak_wolfe
      ɛb = Inf
    end

    if Ar && W
      return (t, t, true, ht, 0.0, 0.0, false)
    end

    iter = 0
    #t=0.0
    #gt=grad(h,t)
    dN=-gt #pour la premiere iteration

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    # le reste de l'algo minimise la fonction φ...
    # par conséquent, le critère d'Armijo sera vérifié φ(t)<φ(0)=0
    φt = φ(t)          # on sait que φ(0)=0

    dφt = dφ(t) # connu dφ(0)=(1.0-τ₀)*g₀
    # Version avec la sécante: modèle quadratique

    if t == 0.0
      A=0.5
      B=0.0
    elseif t == 1.0
      φtprec = 0.0; dφtprec = (1.0-τ₀)*g₀
      s = t - 0.0
      y = dφt - dφtprec
      a=-s
      z= dφt + dφtprec + 3*(φt-φtprec)/a
      discr = z^2-dφtprec*dφt
      denom = dφt + dφtprec + 2*z
      B= 1/3*(dφt+dφtprec+2*z)/(a*a)
      A=-(dφt+z)/a
    end

    if Ar   #version 3
      Δ = 100.0/abs(φt)
    else
      if t == 1.0
        Δ = 1.0/abs(dφt)
      else
        Δ = 1.0/abs(1000.0)
      end
    end

    Quad(d) = φt + dφt*t + A*t^2 + B*t^3 + (1/(4*Δ))*t^4

    t_original = NaN

    verboseLS && println("ɛa = $ɛa ɛb = $ɛb")

    admissible = false
    tired=iter>maxiterLS

    debug && PyPlot.figure(1)
    debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])

    verboseLS && @printf("   iter   t       φt        dφt        Δ        t+d      \n");
    verboseLS && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e\n", iter,t,φt,dφt,Δ,t);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations
        #step computation
        Quad(t) = φt + dφt*t + A*t^2 + B*t^3 + (1/(4*Δ))*t^4
        #dQuad(t) = dφt+2*A*t+3*B*t^2+(1/Δ)*t^3

        #p=Poly([dφt,2*A,3*B,(1/Δ)])

        dR_1=roots([dφt,2*A,3*B,(1/Δ)])

        dR = copy(dR_1)

        vmin=Inf
        for i=1:length(dR)
          rr=dR[i]
          if isfinite(rr) && (imag(rr) < 1e-8)
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

        pred = dφt * d + A * d^2 + B * d^3
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

        if (t + d < 0.0) && (ratio < eps1)  # Unsuccessful
            Δ=red*Δ
            iter=iter+1
            tired=iter>maxiterLS
            verboseLS && @printf("U %4d %9.2e %9.2e  %9.2e %9.2e  %9.2e %9.2e\n", iter,t,φt,dφt,Δ,t+d,φtestTR);
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
                Δ=aug*Δ
            end
            admissible = (dφt>=ɛa) & (dφt<=ɛb)  # Wolfe, Armijo garanti par la
                                                # descente

            if admissible && add_step && (n_add_step < 1)
              n_add_step +=1
              admissible = false
            end

            debug && PyPlot.figure(1)
            debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])
            iter=iter+1
            tired=iter>maxiterLS
            verboseLS && @printf("S %4d %9.2e %9.2e  %9.2e   %9.2e \n", iter,t,φt,dφt,Δ);
        end;
    end;

    # recover h
    ht = φt + h₀ + τ₀*t*g₀
    @assert (t > 0.0) && (!isnan(t)) "invalid step"

    return (t,t_original,true, ht, iter,0,tired)  #pourquoi le true et le 0?

end
