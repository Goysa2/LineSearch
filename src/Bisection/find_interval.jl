export find_interval_ls
# shoud have it's own stopping for finding an interval
function find_interval_ls(h :: LineModel, stop_ls :: LS_Stopping;
                          φ_dφ :: Function = (x, y) -> phi_dphi(x, y),
                          t₀ :: Float64 = 0.0, inc0 :: Float64 = 1.0,
                          τ₀ :: Float64 = 1.0e-4, τ₁ :: Float64 = 0.9999,
                          maxiter :: Int = 100, verboseLS :: Bool = false,
                          kwargs...)

  iter = 1
  inc = inc0
  state = copy(stop_ls.current_state)
  h₀, g₀ = state.h₀, state.g₀

  φt₀ = 0.0                            # φ(0)=0             by definition
  dφt₀ = (1.0 - τ₀) * g₀               # φ''(0)=(1.0-τ₀)*g₀ by definition
  sd = -sign(dφt₀)
  t₁ = t₀ + sd * inc
  update!(state, x = t₁)
  φt1, dφt1 = φ_dφ(h, state; kwargs...)

  ɛa = (τ₁ - τ₀) * g₀
  ɛb = -(τ₁ + τ₀)*g₀

  verboseLS &&
    @printf("  iter t        φt        dφt         t1        φt1        dφt1\n")
  verboseLS &&
    @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n",
            iter, t₀, φt₀, dφt₀, t₁, φt1, dφt1)

  while (dφt1 * sd < 0.0) && (φt1 < φt₀) & (iter < maxiter)
    inc = inc * 4
    t₀ = t₁; φt₀ = φt1; dφt₀ = dφt1
    t₁ = t₀ + sd * inc
    update(state, x = t₁)
    φt1, dφt1 = φ_dφ(h, state; kwargs...)
    iter = iter + 1
    verboseLS && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n",
                         iter, t₀,φt₀,dφt₀,t₁,φt1,dφt1)

  end

  while (dφt1 * sd<0.0) & (iter<maxiter)
    tₘ = (t₁ + t₀) / 2
    update!(state, x = tₘ)
    φₘ, dφₘ = φ_dφ(h, state; kwargs...)
    if φₘ * sd > 0.0
      t₁ = tₘ; φt1 = φₘ; dφt1 = dφₘ
    else
      if φₘ < φt₀
        t₀ = tₘ; φt₀ = φₘ; dφt₀ = dφₘ
      else
        t₁ = tₘ; φt1 = φₘ; dφt1 = dφₘ
      end
    end
    iter = iter + 1
    verboseLS && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", iter, t₀,φt₀,dφt₀,t₁,φt1,dφt1)
  end

  ta = min(t₀, t₁); tb = max(t₀, t₁)

  if ta == t₁
    φta = φt1; dφta = dφt1
    φtb = φt₀; dφtb = dφt₀
  else
    φta = φt₀; dφta = dφt₀
    φtb = φt1; dφtb = dφt1
  end

  return (ta, φta, dφta, tb, φtb, dφtb)

end
