# Stopping manager for LineSearch
export TStopping_LS, start_ls!, stop_ls
type TStopping_LS
  εa :: Float64
  εb :: Float64
  maxiterLS :: Int64
  weak_wolfe :: Bool
  eps1 :: Float64
  eps2 :: Float64
  aug :: Float64
  red :: Float64

  function TStopping_LS(;εa :: Float64 = -maxintfloat(Float64),
                        εb :: Float64 = maxintfloat(Float64),
                        maxiterLS :: Int64 = 50,
                        weak_wolfe :: Bool = false,
                        eps1 :: Float64 = 0.1,
                        eps2 :: Float64 = 0.7,
                        red :: Float64 = 0.15,
                        aug :: Float64 = 10.0,
                        kwargs...)

      return new()
  end
end

function start_ls!(g :: Array{Float64,1},
                   s :: TStopping_LS,
                   τ₀ :: Float64,
                   τ₁ :: Float64,
                   h₀ :: Float64,
                   g₀ :: Float64;
                   maxiterLS :: Int64 = 50,
                   weak_wolfe :: Bool = false,
                   eps1 :: Float64 = 0.1,
                   eps2 :: Float64 = 0.7,
                   red :: Float64 = 0.15,
                   aug :: Float64 = 10.0,
                   kwargs...)
  s.ɛa = (τ₁-τ₀)*g₀
  s.εb = -(τ₁+τ₀)*g₀
  s.maxiterLS = 50; s.red = 0.15; s.aug = 10.0; s.eps1 = 0.1; s.eps2 = 0.7;
  if s.weak_wolfe
    s.εb = Inf
  end
end

function stop_ls(s :: TStopping_LS,
                 dφt:: Float64,
                 iter :: Int64;
                 kwargs...)
  admissible = (dφt >= s.ɛa) & (dφt <= s.ɛb)
  tired = iter > s.maxiterLS

  return admissible, tired
end
