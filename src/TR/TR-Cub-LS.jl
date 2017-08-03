export TR_Cub_ls

# Variation on the classical one dimensionnal trust region method using a two
# point cubic approximation of h. Information about the specific hyper parameters
# are presented in TR_generic_ls

# The scheme is the same but put in different function since it's a different
# approximation

function TR_Cub_ls(h :: LineModel,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   stp_ls :: TStopping_LS = TStopping_LS(),
                   τ₀ :: Float64 = 1.0e-4,
                   τ₁ :: Float64 = 0.9999,
                   verboseLS :: Bool = false,
                   symmetrical :: Bool = false,
                   check_param :: Bool = false,
                   debug :: Bool = false,
                   add_step :: Bool = true,
                   n_add_step :: Int64 = 0,
                   check_slope :: Bool = false,
                   kwargs...)

  (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))
  if check_slope
    (abs(g₀ - grad(h, 0.0)) < 1e-4) || warn("wrong slope")
    verboseLS && @show h₀ obj(h, 0.0) g₀ grad(h, 0.0)
  end

  (t, ht, gt, A_W, Δp, Δn) = init_TR(h, h₀, g₀, g, τ₀, τ₁; kwargs...)

  if A_W
    return (t, t, true, ht, 0.0, 0.0, false)
  end

  t_original = NaN

  iter = 0
  # gt = grad!(h,t,g)
  dN = -gt #pour la premiere iteration

  φ(t) = obj(h, t) - h₀ - τ₀ * t * g₀  # function and
  dφ(t) = grad!(h, t, g) - τ₀ * g₀    # derivative

  start_ls!(g, stp_ls, τ₀, τ₁, h₀, g₀; kwargs...)

  # The rest of the algorithm work with φ
  # therefore, Armijo condition will be satisfied when φ(t)<φ(0)=0
  φt = ht - h₀ - τ₀ * t * g₀
  dφt = gt - τ₀ * g₀
  if t == 0.0
    φt = 0.0              # known that φ(0) = 0.0
    dφt = (1.0 - τ₀) * g₀   # known that φ'(0) = (1.0 - τ₀) * h'(0)
  end

  if t == 0.0
    A=0.5
    B=0.0
  elseif t == 1.0
    φtprec = 0.0; dφtprec = (1.0 - τ₀) * g₀
    s = t - 0.0
    y = dφt - dφtprec
    α = -s
    z = dφt + dφtprec + 3 * (φt - φtprec) / α
    discr = z^2 - dφtprec * dφt
    denom = dφt + dφtprec + 2 * z
    B = 1/3 * (dφt + dφtprec + 2 * z) / (α * α)
    A = -(dφt + z) / α
  end

  # Version avec la sécante: modèle quadratique
  cub(d)= φt + dφt * d + A * d^2 + B * d^3

  verboseLS && @show stp_ls.ɛa, stp_ls.ɛb

  admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)
  verboseLS &&
    @printf("     iter   t       φt        dφt        Δn        Δp    ratio\n");
  verboseLS &&
    @printf("  %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e \n",
            iter, t, φt, dφt, Δn, Δp);

  while !(admissible | tired) #admissible: respecte armijo et wolfe, tired:
                              # nb d'itérations

      if (cub(Δp) < cub(Δn)) || (Δn == 0.0)
        d = Δp
      else
        d = Δn
      end

      # cub(t) = φt + dφt * t + A * t^2 + B * t^3
      # dcub(t) = dφt + 2 * A * t + 3 * B * t^2

      dR = roots([dφt, 2 * A, 3 * B])

      if ((isfinite(dR[1]) && imag(dR[1]) == 0.0) ||
              (isfinite(dR[2]) && imag(dR[2]) == 0.0))
        dR = real(dR)
        if isfinite(dR[1]) && (imag(dR[1]) == 0.0)
          dN2 = dR[1]
        elseif isfinite(dR[2]) && (imag(dR[2]) == 0.0)
          dN2 = dR[2]
        end
        if length(dR) > 1 && ((isfinite(dR[1]) && imag(dR[1]) == 0.0) ||
              (isfinite(dR[2]) && imag(dR[2]) == 0.0))
          if cub(dR[1]) > cub(dR[2])
            dN2 = dR[2]
          end
        end
      else
        dN2 = -dφt * s /y
      end


      if (abs(dN2) < abs(Δp - Δn)) && (cub(d) > cub(dN2))
        d = dN2
      end

      φtestTR = φ(t + d)
      dφtestTR= dφ(t + d)
      # test d'arrêt sur dφ

      pred = dφt * d + A * d^2 + B * d^3;
      if pred > - 1e-10
        ared = (dφt + dφtestTR) * d / 2
      else
        ared = φtestTR - φt;
      end
      ratio = ared / pred

      if ratio < stp_ls.eps1  # Unsuccessful
          Δp = stp_ls.red * Δp
          Δn = stp_ls.red * Δn
          verboseLS &&
            @printf("U %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e  %9e %9e\n",
                    iter, t, φt, dφt, Δn, Δp, d, ratio);
      else             # Successful
          tprec = copy(t)
          φtprec = copy(φt)
          dφtprec = copy(dφt)

          t = t + d

          dφt = copy(dφtestTR)
          φt = copy(φtestTR)

          s = t - tprec
          y = dφt - dφtprec
          α = -s
          z = dφt + dφtprec + 3 * (φt - φtprec) / α
          discr = z^2 - dφtprec * dφt
          denom = dφt + dφtprec + 2 * z
          B = 1/3 * (dφt + dφtprec + 2 * z) / (α * α)
          A = -(dφt + z) / α

          if symmetrical
            if ratio > stp_ls.eps2
              Δp = stp_ls.aug * Δp
              Δn = stp_ls.aug * Δn
            end
          else
            if ratio > stp_ls.eps2
              Δp = stp_ls.aug * Δp
              Δn = min(-t, stp_ls.aug * Δn)
            else
              Δn = min(-t, Δn)
            end
          end

          if admissible && add_step && (n_add_step < 1)
            n_add_step +=1
            admissible = false
          end

          verboseLS &&
            @printf("S %4d %9.2e %9.2e  %9.2e  %9.2e %9.2e  %9e %9e\n",
                    iter, t, φt, dφt, Δn, Δp, d, ratio);
      end;
      iter += 1
      admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)
  end

  # recover h
  ht = φt + h₀ + τ₀ * t * g₀

  t > 0.0 || (verboseLS && @show t dφt )
  @assert (t > 0.0) && (!isnan(t)) "invalid step"
  return (t, t_original, true, ht, iter, 0, tired)

end # function
