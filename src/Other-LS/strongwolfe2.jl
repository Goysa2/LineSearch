export strongwolfe2,zoom2,interpolate2

# TODO: Implement safeguards

# This linesearch algorithm guarantees that the step length
# satisfies the (strong) Wolfe conditions.
# See Nocedal and Wright - Algorithms 3.5 and 3.6

function strongwolfe2(df::AbstractLineFunction,
                      x::Vector,
                      p::Vector,
                      x_new::Vector,
                      gr_new::Vector,
                      lsr::Array{Float64,1},
                      alpha0::Real,
                      mayterminate::Bool;
                      c1::Real = 1e-4,
                      c2::Real = 0.9,
                      rho::Real = 2.0,
                      kwargs...)

    # println(" ")
    # println("c1=",c1," c2=",c2)
    # println("on est dans strongwolfe2 ")
    # println("typeof(df)=",typeof(df))
    # println("x=",x)
    # println("size(x)=",size(x))
    # println("p=",p)
    # println("size(p)=",size(p))
    # println("mayterminate=",mayterminate)
    # println("x_new=",x_new)
    # print("size(x_new)=",size(x_new),"\n")
    # println("gr_new=",gr_new)
    # print("size(gr_new)=",size(gr_new),"\n")
    # println("lsr=",lsr)
    # TODO: do we need gr_new anymore?
    # Any call to gradient! or value_gradient! would update df.g anyway

    # Parameter space
    n = length(x)
    x_return = Array{Float64,1}(n)
    for i=1:n
      x_return[i] = x[i]
    end

    # Step-sizes
    a_0 = 0.0
    a_iminus1 = a_0
    a_i = alpha0
    a_max = 65536.0

    # phi(alpha) = df.f(x + alpha * p)
    phi_0 = lsr[1]
    phi_a_iminus1 = phi_0
    phi_a_i = NaN

    # phi'(alpha) = vecdot(g(x + alpha * p), p)
    phiprime_0 = lsr[2]
    # phiprime_0 = g₀
    phiprime_a_i = NaN

    # Iteration counter
    i = 1

    while a_i < a_max
        # Update x_new
        # @simd for index in 1:n
        #     @inbounds x_new[index] = x[index] + a_i * p[index]
        # end

        # println("après update x_new x=",x)

        # Evaluate phi(a_i)
        phi_a_i = obj(df, a_i)

        # Test Wolfe conditions
        if (phi_a_i > phi_0 + c1 * a_i * phiprime_0) ||
            (phi_a_i >= phi_a_iminus1 && i > 1)
            #println("on rentre dans zoom2 1")
            a_star = zoom2(a_iminus1, a_i,
                          phiprime_0, phi_0,
                          df, x, p, x_new, gr_new)
            x = x_return
            # println("en sortant de strongwolfe2 x_return=",x_return)
            # println("en sortant de strongwolfe2 x=",x)
            # println("en sortant de strongwolfe2 x_new=",x_new)
            return a_star, false, x_new, i, 0.0
        end

        # Evaluate phi'(a_i)
        # NLSolversBase.gradient!(df, x_new)
        # gr_new[:] = gradient(df)
        #
        # phiprime_a_i = vecdot(gr_new, p)

        phiprime_a_i = grad(df, a_i)

        # Check condition 2
        if abs(phiprime_a_i) <= -c2 * phiprime_0
            x = x_return
            #println("on fini strongwolfe2 sans zoom2")
            # println("en sortant de strongwolfe2 x=",x)
            # println("en sortant de strongwolfe2 x_new=",x_new)
            return a_i, false, x_new, i, 0.0
        end

        # Check condition 3
        if phiprime_a_i >= 0.0
            #println("on rentre dans zoom2 2")
            a_star = zoom2(a_i, a_iminus1,
                          phiprime_0, phi_0,
                          df, x, p, x_new, gr_new)
            x = x_return
            # println("en sortant de strongwolfe2 x=",x)
            # println("en sortant de strongwolfe2 x_new=",x_new)
            return a_star, false, x_new, i, 0.0
        end

        # Choose a_iplus1 from the interval (a_i, a_max)
        a_iminus1 = a_i
        a_i *= rho

        # Update phi_a_iminus1
        phi_a_iminus1 = phi_a_i

        # Update iteration count
        i += 1
    end

    # Quasi-error response
    x = x_return
    # println("en sortant de strongwolfe2 x=",x)
    # println("en sortant de strongwolfe2 x_new=",x_new)
    return a_max, false, x_new, i, 0.0
end

function zoom2(a_lo::Real,
               a_hi::Real,
               phiprime_0::Real,
               phi_0::Real,
               df :: C1LineFunction,
               x::Vector,
               p::Vector,
               x_new::Vector,
               gr_new::Vector;
               c1::Real = 1e-4,
               c2::Real = 0.9,
               kwargs...)

    #println("on est dans zoom2")
    #println("a_lo=",a_lo," a_hi=",a_hi)
    # Parameter space
    n = length(x)

    # Step-size
    a_j = NaN

    # Count iterations
    iteration = 0
    max_iterations = 10

    # Shrink bracket
    while iteration < max_iterations
        iteration += 1

        # Cache phi_a_lo
        # @simd for index in 1:n
        #     @inbounds x_new[index] = x[index] + a_lo * p[index]
        # end
        # phi_a_lo = NLSolversBase.value_gradient!(df, x_new)
        phi_a_lo = obj(df, a_lo)
        # gr_new[:] = NLSolversBase.gradient(df)
        # phiprime_a_lo = vecdot(gr_new, p)
        phiprime_a_lo = grad(df, a_lo)
        # Cache phi_a_hi
        # @simd for index in 1:n
        #     @inbounds x_new[index] = x[index] + a_hi * p[index]
        # end
        # phi_a_hi = NLSolversBase.value_gradient!(df, x_new)
        phi_a_hi = obj(df, a_hi)
        # gr_new[:] = NLSolversBase.gradient(df)
        # phiprime_a_hi = vecdot(gr_new, p)

        phiprime_a_hi = grad(df, a_hi)

        # Interpolate a_j
        if a_lo < a_hi
            a_j = interpolate2(a_lo, a_hi,
                              phi_a_lo, phi_a_hi,
                              phiprime_a_lo, phiprime_a_hi)
        else
            # TODO: Check if this is needed
            a_j = interpolate2(a_hi, a_lo,
                              phi_a_hi, phi_a_lo,
                              phiprime_a_hi, phiprime_a_lo)
        end

        # println("iteration=",iteration," a_j=",a_j)

        # Update x_new
        # @simd for index in 1:n
        #     @inbounds x_new[index] = x[index] + a_j * p[index]
        # end

        # Evaluate phi(a_j)
        # phi_a_j = NLSolversBase.value!(df, x_new)
        phi_a_j = obj(df, a_j)

        # Check Armijo
        if (phi_a_j > phi_0 + c1 * a_j * phiprime_0) ||
            (phi_a_j > phi_a_lo)
            a_hi = a_j
        else
            # Evaluate phiprime(a_j)
            # NLSolversBase.gradient!(df, x_new)
            # gr_new[:] = gradient(df)
            # phiprime_a_j = vecdot(gr_new, p)
            phiprime_a_j = grad(df, a_j)

            if abs(phiprime_a_j) <= -c2 * phiprime_0
                return a_j
            end

            if phiprime_a_j * (a_hi - a_lo) >= 0.0
                a_hi = a_lo
            end

            a_lo = a_j
        end
    end

    # if a_lo==a_hi
    #   warn("t_lo==a_hi")
    # end

    # Quasi-error response
    return a_j
end

# a_lo = a_{i - 1}
# a_hi = a_{i}
function interpolate2(a_i1::Real, a_i::Real,
                     phi_a_i1::Real, phi_a_i::Real,
                     phiprime_a_i1::Real, phiprime_a_i::Real,kwargs...)
    # println("on est dans interpolate2")
    # println("a_i1=",a_i1," a_i=",a_i)
    # println("phi_a_i1=",phi_a_i1," phi_a_i=",phi_a_i)
    # println("phiprime_a_i1=",phiprime_a_i1," phiprime_a_i=",phiprime_a_i)
    d1 = phiprime_a_i1 + phiprime_a_i -
        3.0 * (phi_a_i1 - phi_a_i) / (a_i1 - a_i)
    if (d1 * d1 - phiprime_a_i1 * phiprime_a_i) < 0.0
     warn("Interpolation with a negative square root")
     return NaN
    end
    d2 = sqrt(d1 * d1 - phiprime_a_i1 * phiprime_a_i)
    #println("dN=",- (a_i - a_i1) *((phiprime_a_i + d2 - d1) /(phiprime_a_i - phiprime_a_i1 + 2.0 * d2)))
    return a_i - (a_i - a_i1) *
        ((phiprime_a_i + d2 - d1) /
         (phiprime_a_i - phiprime_a_i1 + 2.0 * d2))
end
