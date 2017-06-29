export TR_Cub_ls
function TR_Cub_ls(h :: AbstractLineFunction2,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   τ₀ :: Float64=1.0e-4,
                   eps1 :: Float64 = 0.1,
                   eps2 :: Float64 = 0.7,
                   red :: Float64 = 0.15,
                   aug :: Float64= 10.0,
                   τ₁ :: Float64=0.9999,
                   maxiterLS :: Int64=50,
                   verboseLS :: Bool=false,
                   check_param :: Bool = false,
                   borne_inf :: Float64 = -10.0,
                   borne_sup :: Float64 = 10.0,
                   print_cub :: Bool = false,
                   debug :: Bool = false,
                   a :: Float64 = 0.0,
                   b :: Float64 = 25.0,
                   print_cub_iter :: Int64 = 3,
                   add_step :: Bool = true,
                   kwargs...)

    (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))

    (t,ht,gt,A_W,Δp,Δn,ɛa,ɛb)=init_TR(h,h₀,g₀,g,τ₀,τ₁;kwargs...)

    if A_W
      return (t, t, true, ht, 0.0, 0.0, false, h.f_eval, h.g_eval, h.h_eval)
    end

    t_original = NaN

    iter = 0
    gt=grad(h,t)
    dN=-gt #pour la premiere iteration

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    if verboseLS && print_cub && (iter == print_cub_iter)
      i = a
      x_axis = []
      y_axis = []
      while i <= b              #"shape" of φ
        push!(x_axis, i)
        push!(y_axis, φ(i))
        i += 1.0
      end
      PyPlot.figure(4)
      PyPlot.scatter(x_axis,y_axis)
    end

    # le reste de l'algo minimise la fonction φ...
    # par conséquent, le critère d'Armijo sera vérifié φ(t)<φ(0)=0
    φt = φ(t)          # on sait que φ(0)=0

    dφt = dφ(t) # connu dφ(0)=(1.0-τ₀)*g₀

    if t == 0.0
      A=0.5
      B=0.0
    elseif t == 1.0
      φtprec = 0.0; dφtprec = (1.0-τ₀)*g₀
      s = t - 0.0
      y = dφt - dφtprec
      α=-s
      z= dφt + dφtprec + 3*(φt-φtprec)/α
      discr = z^2-dφtprec*dφt
      denom = dφt + dφtprec + 2*z
      B= 1/3*(dφt+dφtprec+2*z)/(α*α)
      A=-(dφt+z)/α
    end

    n_add_step = 0


    # Version avec la sécante: modèle quadratique
    cub(d)= φt + dφt*d + A*d^2 + B*d^3

    verboseLS && @show ɛa, ɛb

    admissible = false
    t_original = NaN
    tired=iter > maxiterLS
    verboseLS && @printf("     iter   t       φt        dφt        Δn        Δp            d            ratio\n");
    verboseLS && @printf("  %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e \n", iter,t,φt,dφt,Δn,Δp);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations


        if (cub(Δp)<cub(Δn)) | (Δn==0.0)
          d=Δp
        else
          d=Δn
        end

        # verboseLS

        cub(t) = φt + dφt*t + A*t^2 + B*t^3
        dcub(t) = dφt + 2 * A * t + 3 * B * t^2

        print_cub && verboseLS && @show φt dφt A B

        if print_cub && (iter == print_cub_iter)
          x1 = PyPlot.linspace(borne_inf, borne_sup, 200)
          y1 = φt + dφt*x1 + A*x1.^2 + B*x1.^3
          # PyPlot.clf()
          PyPlot.figure(1)
          PyPlot.plot(x1,y1)
        end

        if print_cub && (iter == print_cub_iter)
          x2 = PyPlot.linspace(borne_inf, borne_sup, 200)
          y2 = dφt + 2 * A * x2 + 3 * B * x2.^2
          PyPlot.figure(2)
          PyPlot.plot(x2, y2)
        end


        #dcub(t)=dφt + 2*A*t + 3*B*t^2
        #p=Poly([dφt,2*A,3*B])
        #dR=roots(p)                        #for now the roots tool is unstable

        dR = []

        discr_pol = (2*A)^2 - 4 * (3*B) * dφt

        if B != 0
          if discr == 0
            root = (-2 * A)/(6 * B)
            push!(dR, root)
          elseif discr > 0
            if A < 0
              root1 = (2 * dφt)/(-2 * A + sqrt(discr_pol))
              root2 = (-2 * A + sqrt(discr_pol))/(6 * B)
            else
              root1 = (-2 * A - sqrt(discr_pol))/(6 * B)
              root2 = (2 * dφt)/(-2 * A - sqrt(discr_pol))
            end
            push!(dR, root1)
            push!(dR, root2)
          end
        elseif B == 0.0
          root = (-dφt)/2*A
          push!(dR, root)
        end

        print_cub && verboseLS && @show dR


        if !isempty(dR) && (isfinite(dR[1]) || isfinite(dR[2]))        ###isreal(dR) & !isempty(dR)
          dR=real(dR)
          if isfinite(dR[1])
            dN2=dR[1]
          elseif isfinite(dR[2])
            dN2 = dR[2]
          end
          if length(dR)>1 && (isfinite(dR[1])) && (isfinite(dR[2]))
            if cub(dR[1])>cub(dR[2])
              dN2=dR[2]
            end
          end
        else
          print_cub && verboseLS && @show dφt s y
          dN2=-dφt*s/y
        end

        print_cub && verboseLS && @show dN2
        print_cub && verboseLS && @show cub(d) cub(dN2)

        if (abs(dN2)<abs(Δp-Δn)) & (cub(d)>cub(dN2))
          d=dN2
        end

        print_cub && verboseLS && @show d

        φtestTR = φ(t+d)
        dφtestTR= dφ(t+d)
        # test d'arrêt sur dφ

        # (pred,ared,ratio)=pred_ared_computation(dφt,φt,ddφt,d,φtestTR,dφtestTR)
        pred = dφt*d + A*d^2 + B * d^3;
        if pred > - 1e-10
          ared = (dφt + dφtestTR)*d/2
        else
          ared = φtestTR-φt;
        end
        ratio = ared / pred

        if ratio < eps1  # Unsuccessful
            Δp = red*Δp
            Δn = red*Δn
            verboseLS && @printf("U %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e  %9e  %9e\n", iter,t,φt,dφt,Δn,Δp,d,ratio);
        else             # Successful
            tprec = t
            φtprec =φt
            dφtprec = dφt

            print_cub && verboseLS && @show t, d
            t = t + d
            print_cub && verboseLS && @show t
            dφt = dφtestTR
            φt = φtestTR

            print_cub && verboseLS && @show φtestTR, dφtestTR

            s = t-tprec
            y = dφt - dφtprec
            α=-s
            z= dφt + dφtprec + 3*(φt-φtprec)/α
            discr = z^2-dφtprec*dφt
            denom = dφt + dφtprec + 2*z
            B= 1/3*(dφt+dφtprec+2*z)/(α*α)
            A=-(dφt+z)/α

            #verboseLS && @show s y α discr denom B A

            # if ratio > eps2
            #     Δp = aug * Δp
            #     Δn = min(-t, aug * Δn)
            # else
            #     Δn = min(-t, Δn)
            # end
            if ratio >eps2     #numerical tests indicates this condition improves the performance
              Δp = aug * Δp
              Δn = aug * Δn
            end

            admissible = (dφt>=ɛa) & (dφt<=ɛb)  # Wolfe, Armijo garanti par la
                                                # descente

            if admissible && add_step && (n_add_step < 1)
              n_add_step +=1
              admissible = false
            end


            verboseLS && @printf("S %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e  %9e  %9e\n", iter,t,φt,dφt,Δn,Δp,d, ratio);
        end;
        iter+=1
        tired=iter>maxiterLS
    end

    # recover h
    ht = φt + h₀ + τ₀*t*g₀
    return (t, t_original,true, ht, iter,0,tired, h.f_eval, h.g_eval, h.h_eval)   #pourquoi le true et le 0?

end
