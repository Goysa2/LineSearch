function Armijo(t::Float64,
                ht::Float64,
                gt::Float64,
                h₀::Float64,
                g₀::Float64,
                τ₀::Float64)
    # Hager & Zhang numerical trick
    fact = -0.8
    Eps = 1e-10
    hgoal = h₀ + g₀ * t * τ₀
    Armijo_HZ = (ht <= hgoal) | ((ht <= h₀ + Eps * abs(h₀)) & (gt <= fact * g₀)) 
    return Armijo_HZ
end
