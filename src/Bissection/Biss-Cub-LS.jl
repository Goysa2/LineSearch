export Biss_Cub_ls
function Biss_Cub_ls(h :: LineModel,
                     h₀ :: Float64,
                     g₀ :: Float64,
                     g :: Array{Float64,1};
                     γ :: Float64=0.8,
                     τ₀ :: Float64=1.0e-4,
                     τ₁ :: Float64=0.9999,
                     stp_ls :: TStopping_LS = TStopping_LS(),
                     verboseLS :: Bool=false,
                     check_param :: Bool = false,
                     check_slope :: Bool = false,
                     debug :: Bool = false,
                     add_step :: Bool = true,
                     n_add_step :: Int64 = 0,
                     weak_wolfe :: Bool = false,
                     kwargs...)

 (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))

 if check_slope
   (abs(g₀ - grad(h, 0.0)) < 1e-4) || warn("wrong slope")
   verboseLS && @show h₀ obj(h, 0.0) g₀ grad(h,0.0)
 end

 t = 1.0
 ht = obj(h,t)
 gt = grad!(h, t, g)
 if Armijo(t,ht,gt,h₀,g₀,τ₀) && Wolfe(gt,g₀,τ₁)
   return (t, t, true, ht, 0, 0, false)
 end

 #println("au début de Biss_Cub_ls g₀=",g₀)

 (ta, φta, dφta, tb, φtb, dφtb) = find_interval_ls(h, h₀, g₀, g; kwargs...)

 #println("on est après le find_interval_ls")
 t=tb
 tp=ta
 tqnp=ta
 iter=0

 φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
 dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

 start_ls!(g, stp_ls, τ₀, τ₁, h₀, g₀; kwargs...)


 iter = 0

 φt = φta
 φtm1 = φtb
 dφt = dφta
 dφtm1 = dφtb

 dφa = dφta
 dφb = dφtb

 admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)
 t_original = NaN

 debug && PyPlot.figure(1)
 debug && PyPlot.scatter([t],[φt + h₀ + τ₀ * t * g₀])

 verboseLS && @printf("iter        tqnp        t         dφtm1        dφt \n")
 verboseLS && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e\n",
                      iter, tqnp, t, dφtm1, dφt)

 while !(admissible | tired)

   s = t - tqnp
   y = dφt - dφtm1

   α = -s
   z = dφt + dφtm1 + 3 * (φt - φtm1) / α
   discr = z^2 - dφt * dφtm1
   denom = dφt + dφtm1 + 2 * z
   if (discr > 0.0) && (abs(denom) > eps(Float64))
     #si on peut on utilise l'interpolation cubique
     w = sqrt(discr)
     dN = -s * (dφt + z + sign(α) * w) / (denom)
   else #on se rabat sur une étape de sécante
     dN = -dφt * s / y
   end

   if ((tp-t) * dN > 0.0) && (dN / (tp - t) < γ)
     tplus = t + dN
     φplus = φ(tplus)
     dφplus = dφ(tplus)
     verboseLS && println("N")
   else
     tplus = (t + tp) / 2
     φplus = φ(tplus)
     dφplus = dφ(tplus)
     verboseLS && println("B")
   end

   if t > tp
     if dφplus < 0.0
       tp = t
       tqnp = t
       t = tplus
     else
       tqnp = t
       t = tplus
     end
   else
     if dφplus > 0.0
       tp = t
       tqnp = t
       t = tplus
     else
       tqnp = t
       t = tplus
     end
   end
   #println("tp=",tp," t=",t)
   φtm1 = φt
   dφtm1 = dφt
   φt = φplus
   dφt = dφplus

   iter += 1
   admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)

   if admissible && add_step && (n_add_step < 1)
     t_original = copy(t)
     n_add_step += 1
     admissible = false
   end

   debug && PyPlot.figure(1)
   debug && PyPlot.scatter([t],[φt + h₀ + τ₀ * t * g₀])

   verboseLS && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e\n",
                        iter, tqnp, t, dφtm1, dφt)
 end

 ht = φt + h₀ + τ₀ * t * g₀

 @assert (t > 0.0) && (!isnan(t)) "invalid step"

 return (t, t_original, true, ht, iter, 0, tired)
end
