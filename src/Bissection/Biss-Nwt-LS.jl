export Biss_Nwt_ls
function Biss_Nwt_ls(h :: LineModel,
                     h₀ :: Float64,
                     g₀ :: Float64,
                     g :: Array{Float64,1};
                     γ :: Float64 = 0.8,
                     τ₀ :: Float64 = 1.0e-4,
                     τ₁ :: Float64 = 0.9999,
                     stp_ls :: TStopping_LS = TStopping_LS(),
                     verboseLS :: Bool = false,
                     check_param :: Bool = false,
                     debug :: Bool = false,
                     add_step :: Bool = true,
                     n_add_step :: Int64 = 0,
                     check_slope :: Bool = false,
                     weak_wolfe :: Bool = false,
                     kwargs...)

  (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))
  if check_slope
    (abs(g₀ - grad(h, 0.0)) < 1e-4) || warn("wrong slope")
    verboseLS && @show h₀ obj(h, 0.0) g₀ grad(h, 0.0)
  end

  t = 1.0
  ht = obj(h, t)
  gt = grad!(h, t, g)
  Ar = Armijo(t, ht, gt, h₀, g₀, τ₀); Wo = Wolfe(gt, g₀, τ₁)
  if Ar && Wo
    return (t, t, false, ht, 0, 0, false)
  end

  (ta, φta, dφta, tb, φtb, dφtb) = find_interval_ls(h, h₀, g₀, g,
                                                    verboseLS = verboseLS,
                                                    debug = debug; kwargs...)
  verboseLS && println("ta = $ta tb = $tb")

   t = ta
   tp = tb
   tqnp = tb
   iter = 0

   φ(t) = obj(h, t) - h₀ - τ₀ * t * g₀  # function and
   dφ(t) = grad!(h, t, g) - τ₀ * g₀     # derivative
   ddφ(t) = hess(h, t)

   iter = 0

   dφt = dφta
   dφa = dφta
   dφb = dφtb

   start_ls!(g, stp_ls, τ₀, τ₁, h₀, g₀; kwargs...)

   verboseLS && println("ϵₐ = $(stp_ls.ɛa) ϵᵦ = $(stp_ls.ɛb)")

   admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)
   t_original = NaN

   debug && PyPlot.figure(1)
   debug && PyPlot.scatter([t],[φ(t) + h₀ + τ₀ * t * g₀])

   verboseLS && @printf(" iter     tqnp        t         dφtqnp     dφt \n")
   verboseLS && @printf(" %7e %7.2e  %7.2e  %7.2e  %7.2e\n",
                        iter, tqnp, t, dφa, dφb)

   while !(admissible | tired) # admissible: satisfies Armijo & Wolfe,
                               # tired: exceeds maximum number of iterations
     ddφt = ddφ(t)
     dN = -dφt / ddφt

     if ((tp - t) * dN > 0.0) && (dN / (tp - t) < γ)
       tplus = t + dN
       #hplus = φ(tplus)
       dφplus = dφ(tplus)
       verboseLS && println("N")
     else
       tplus = (t + tp) / 2
       #hplus = φ(tplus)
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

     dφtqnp = dφt

     dφt = dφplus

     iter += 1
     admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)
     if admissible && add_step && (n_add_step < 1)
       t_original = copy(t)
       n_add_step += 1
       admissible = false
     end

     debug && PyPlot.figure(1)
     debug && PyPlot.scatter([t], [φ(t) + h₀ + τ₀ * t * g₀])

     verboseLS && @printf(" %7e %7.2e  %7.2e  %7.2e  %7.2e\n",
                          iter, tqnp, t, dφtqnp, dφt)
   end

   ht = φ(t) + h₀ + τ₀ * t * g₀

   #@assert (t > 0.0) && (!isnan(t)) "invalid step"

   return (t, t_original, false, ht, iter, 0, tired)

end
