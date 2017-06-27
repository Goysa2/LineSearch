export TR_Cub_ls
function TR_Cub_ls(h :: AbstractLineFunction2,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   τ₀ :: Float64=1.0e-4,
                   eps1 :: Float64 = 0.2,
                   eps2 :: Float64 = 0.8,
                   red :: Float64 = 0.15,
                   aug :: Float64= 10.0,
                   τ₁ :: Float64=0.9999,
                   maxiterLS :: Int64=50,
                   verboseLS :: Bool=false,
                   check_param :: Bool = false,
                   kwargs...)

    (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))

    (t,ht,gt,A_W,Δp,Δn,ɛa,ɛb)=init_TR(h,h₀,g₀,g,τ₀,τ₁)

    if A_W
      return (t, t, true, ht, 0.0, 0.0, false, h.f_eval, h.g_eval, h.h_eval)
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


    admissible = false
    t_original = NaN
    tired=iter > maxiterLS
    verboseLS && @printf("   iter   t       φt        dφt        Δn        Δp  \n");
    verboseLS && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e \n", iter,t,φt,dφt,Δn,Δp);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations

        if (q(Δp)<q(Δn)) | (Δn==0.0)
            d=Δp
        else
            d=Δn
        end

        cub(t)= φt + dφt*t + A*t^2 + B*t^3
        #dcub(t)=dφt + 2*A*t + 3*B*t^2
        #p=Poly([dφt,2*A,3*B])
        #dR=roots(p)                        #for now the roots tool is unstable


        # dR = PolynomialRoots.roots([dφt, 2*A, 3*B])
        # #dR_1 = Complex{Float64}[NaN+NaN*im,3.24-0.0im]

        dR = []

        discr = (2*A)^2 - 4 * (3*B) * dφt

        if B != 0
          if discr == 0
            root = (-2 * A)/(6 * B)
            push!(dR, root)
          elseif discr > 0
            root1 = (-2 * A - sqrt(discr))/(6 * B)
            root2 = (-2 * A + sqrt(discr))/(6 * B)
            push!(dR, root1)
            push!(dR, root2)
          end
        elseif B == 0.0
          root = (-dφt)/2*A
          push!(dR, root)
        end

        verboseLS && @show dR


        if !isempty(dR)         ###isreal(dR) & !isempty(dR)
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
          dN2=-dφt*s/y
        end

        if (abs(dN2)<abs(Δp-Δn)) & (q(d)>q(dN2))
          d=dN2
        end

        verboseLS && @show d

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
            verboseLS && @printf("U %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e  %9.2e %9.2e\n", iter,t,φt,dφt,Δn,Δp,t+d,φtestTR);
        else             # Successful
            tprec = t
            φtprec =φt
            dφtprec = dφt


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


            verboseLS && @printf("S %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e \n", iter,t,φt,dφt,Δn,Δp);
        end;
        iter+=1
        tired=iter>maxiterLS
    end

    # recover h
    ht = φt + h₀ + τ₀*t*g₀
    return (t, t_original,true, ht, iter,0,tired, h.f_eval, h.g_eval, h.h_eval)   #pourquoi le true et le 0?

end
