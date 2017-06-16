function Wolfe(gt::Float64,
               g₀::Float64,
               τ₁::Float64)

    wolfe = (abs(gt) <= -τ₁*g₀)
    return wolfe
end
