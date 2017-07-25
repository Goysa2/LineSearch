export Armijo

# Function designed to test if a function h(t) = f(x + t⋆d) satisfies the Armijo
# condition at a given point t. The inputs required are:
# t :: the point at which we want to test if the Armijo condition is satisfied
# ht :: h(t)
# gt :: g(t) = h'(t)
# h₀ :: h(0)
# g₀ :: g(0) = h'(0)
# τ₀ :: The parameter associated with the Armijo condition
#
# The function returns true if Armijo condition is satisfied. We use the standard
# Armijo condition enriched with the Hager & Zhang numerical trick for the
# condition

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
    Armijo_HZ = (ht <= hgoal) || ((ht <= h₀ + Eps * abs(h₀)) & (gt <= fact * g₀))
    return Armijo_HZ
end
