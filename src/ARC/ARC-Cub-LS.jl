export ARC_Cub_ls

# In this version of ARC our approximation of h is now:
# q(d) = h(θ) + h'(θ)⋆d +  A ⋆ d² + B ⋆ d³ + (1/Δ)*|d|⁴.
# The scheme is the same but put in different function since it's a different
# approximation


function ARC_Cub_ls(h :: LineModel,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   stp_ls :: TStopping_LS = TStopping_LS(),
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
    verboseLS && @show h₀ obj(h, 0.0) g₀ grad(h, 0.0)
  end

  (t, ht, gt, Ar, W) = init_ARC(h, h₀, g₀, g, τ₀, τ₁)

  if Ar && W
    return (t, t, true, ht, 0.0, 0.0, false)
  end

  iter = 0
  dN = -gt #pour la premiere iteration

  φ(t) = obj(h, t) - h₀ - τ₀ * t * g₀  # fonction et
  dφ(t) = grad!(h, t, g) - τ₀ * g₀    # dérivée

  start_ls!(g, stp_ls, τ₀, τ₁, h₀, g₀; kwargs...)

  # le reste de l'algo minimise la fonction φ...
  # par conséquent, le critère d'Armijo sera vérifié φ(t)<φ(0)=0
  φt = ht - h₀ - τ₀ * t * g₀
  dφt = gt - τ₀ * g₀
  if t == 0.0
      φt = 0.0
      dφt = (1 - τ₀) * g₀
  end

  if t == 0.0
    A = 0.5
    B = 0.0
  elseif t == 1.0
    φtprec = 0.0; dφtprec = (1.0 - τ₀) * g₀
    s = t - 0.0
    y = dφt - dφtprec
    a = -s
    z = dφt + dφtprec + 3 * (φt - φtprec) / a
    discr = z^2 - dφtprec * dφt
    denom = dφt + dφtprec + 2*z
    B = 1/3 * (dφt + dφtprec + 2 * z) / (a * a)
    A =-(dφt + z) / a
  end

  if Ar   #version 3              # TODO: find a better way to select Δ
    Δ = 100.0 / abs(φt)           # Also in ARC_generic_ls
  else
    if t == 1.0
      Δ = 1.0 / abs(dφt)
    else
      Δ = 1.0 / abs(1000.0)
    end
  end

  Quad(d) = φt + dφt * t + A * t^2 + B * t^3 + (1 / (4 * Δ)) * t^4

  t_original = NaN

  verboseLS && println("ɛa = $(stp_ls.ɛa) ɛb = $(stp_ls.ɛb)")

  admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)

  debug && PyPlot.figure(1)
  debug && PyPlot.scatter([t], [φt + h₀ + τ₀ * t * g₀])

  verboseLS && @printf("  iter   t       φt        dφt        Δ        t+d \n");
  verboseLS && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e\n",
                       iter, t, φt, dφt, Δ, t);

  while !(admissible | tired) # admissible: respecte armijo et wolfe
                              # tired: nb d'itérations
      # direction computation
      # Quad(t) = φt + dφt * t + A * t^2 + B * t^3 + (1 / (4 * Δ)) * t^4
      # Quad'(t) = dφt + 2 * A * t + 3 * B * t² + (1 / Δ) * t³

      dR = roots(Poly([dφt, 2 * A, 3 * B, (1 / Δ)]))

      vmin = Inf
      for i = 1:length(dR)
        rr = dR[i]
        if isfinite(rr) && (imag(rr) < 1e-8)
          rr = real(rr)
          vact = Quad(rr)
          if rr * dφt < 0.0
            if vact < vmin
              dN = rr
              vmin =  vact
            end
          end
        end
      end

      d = dN

      if d == 0.0
        ht = φt + h₀ + τ₀ * t * g₀
        return (t, true, ht, iter + 1 , 0, true)
      end


      φtestTR = φ(t + d)
      dφtestTR = dφ(t + d)
      # test d'arrêt sur dφ

      pred = dφt * d + A * d^2 + B * d^3
      #assert(pred<0)   # How to recover? As is, it seems to work...
      if pred > -1e-10
        ared = (dφt + dφtestTR) * d / 2
      else
        ared = φtestTR - φt
      end

      tprec = t;
      φtprec = φt;
      dφtprec = dφt;
      ratio = ared / pred

      if (t + d < 0.0) && (ratio < stp_ls.eps1)  # Unsuccessful
          Δ = stp_ls.red * Δ
          verboseLS &&
            @printf("U %4d %9.2e %9.2e  %9.2e %9.2e  %9.2e %9.2e\n",
                    iter, t, φt, dφt, Δ, t+d, φtestTR);
      else             # Successful
          t = t + d

          dφt = dφtestTR
          φt = φtestTR

          s = t-tprec
          y = dφt- dφtprec

          a = -s
          z = dφt + dφtprec + 3 * (φt - φtprec) / a
          discr = z^2 - dφt * dφtprec
          denom = dφt + dφtprec + 2 * z
          B = (1/3) * (dφt + dφtprec + 2 * z) / (a * a)
          A = -(dφt + z) / a

          if ratio > stp_ls.eps2
              Δ = stp_ls.aug * Δ
          end

          if admissible && add_step && (n_add_step < 1)
            n_add_step += 1
            admissible = false
          end

          debug && PyPlot.figure(1)
          debug && PyPlot.scatter([t], [φt + h₀ + τ₀ * t * g₀])
          verboseLS && @printf("S %4d %9.2e %9.2e  %9.2e   %9.2e \n",
                                iter, t, φt, dφt, Δ);
      end;
      iter += 1
      admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)
  end;

  # recover h
  ht = φt + h₀ + τ₀ * t * g₀
  #@assert (t > 0.0) && (!isnan(t)) "invalid step"

  return (t, t_original, true, ht, iter, 0, tired)
end # function
