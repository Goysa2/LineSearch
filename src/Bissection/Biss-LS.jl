export Biss_ls

# generic bissection algorithm designed to return an admissible step size.

function Biss_ls(h :: LineModel,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 g :: Array{Float64,1};
                 τ₀ :: Float64 = 1.0e-4,
                 τ₁ :: Float64 = 0.9999,
                 stp_ls :: TStopping_LS = TStopping_LS(),
                 verboseLS :: Bool = false,
                 check_param :: Bool = false,
                 check_slope :: Bool = false,
                 add_step :: Bool = true,
                 n_add_step :: Int64 = 0,
                 weak_wolfe :: Bool = false,
                 kwargs...)

  (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))
  if check_slope
    (abs(g₀ - grad(h, 0.0)) < 1e-4) || warn("wrong slope")
    verboseLS && @show h₀ obj(h, 0.0) g₀ grad(h,0.0)
  end

  # We check if the step size of one is admissible
  t = 1.0
  ht = obj(h, t)
  gt = grad!(h, t, g)
  if Armijo(t, ht, gt, h₀, g₀, τ₀) && Wolfe(gt, g₀, τ₁)
    return (t, t, true, ht, 0, 0, false)
  end

  # We find an interval containing an admissible step size
  (ta, φta, dφta, tb, φtb, dφtb) = find_interval_ls(h, h₀, g₀, g; kwargs...)

  φ(t) = obj(h,t) - h₀ - τ₀ * t * g₀  # function and
  dφ(t) = grad!(h, t, g) - τ₀ * g₀    # derivative

  start_ls!(g, stp_ls, τ₀, τ₁, h₀, g₀; kwargs...)

  # We cut the interval in 2
  tp = (ta + tb) / 2

  iter = 0

  admissible, tired = stop_ls(stp_ls, dφ(t), iter; kwargs...)
  t_original = NaN
  verboseLS &&
    @printf("   iter   ta       tb        tp        dφp\n");
  verboseLS &&
    @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e \n", iter, ta, tb, tp, NaN)

  while !(admissible | tired) # admissible: satisfies Armijo & Wolfe,
                              # tired: nb of iterations

    if iter > 0               # We don't want to compute tp twice
      tp = (ta + tb) / 2
    end

    dφp = dφ(tp)

    # We adjust our intervals depending on the sign of φ'(tp)
    if dφp <= 0.0
      ta = tp
      dφa = dφp
    else
      tb = tp
      dφb = dφp
    end

    iter += 1
    admissible, tired = stop_ls(stp_ls, dφp, iter; kwargs...)

    # Additional step if desired
    if admissible && add_step && (n_add_step < 1)
      n_add_step +=1
      admissible = false
    end

    verboseLS && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e\n",
                          iter, ta, tb, tp, dφp);
  end;

  ht = φ(tp) + h₀ + τ₀ * tp * g₀

  @assert (t > 0.0) && (!isnan(t)) "invalid step"

  return (tp, t_original, true, ht, iter, 0, tired)
end
