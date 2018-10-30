export Wolfe

# Function designed to test if a function h(t) = f(x + t⋆d) satisfies the Wolfe
# condition at a given point t. The inputs required are:
# gt :: g(t) = h'(t)
# g₀ :: g(0) = h'(0)
# τ₁ :: parameter associated with the Wolfe condition
#
# returns true if the Wolfe condition is satisfied

function Wolfe(gt::Float64,
               g₀::Float64,
               τ₁::Float64)

    wolfe = (abs(gt) <= -τ₁*g₀)
    return wolfe
end
