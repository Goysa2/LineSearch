export strongwolfe2,zoom2,interpolate2

# TODO: Implement safeguards

# This linesearch algorithm guarantees that the step length
# satisfies the (strong) Wolfe conditions.
# See Nocedal and Wright - Algorithms 3.5 and 3.6

function strongwolfe2(df :: AbstractLineFunction,
                      #x::Vector,
                      h₀ :: Float64,
                      g₀ :: Float64,
                      g :: Array{Float64,1};
                      # p::Vector,                  #équivalent de d donc ce linesearch à besoin de d à un certain point
                      # x_new::Vector,
                      # gr_new::Vector,
                      # lsr::LineSearchResults{T},
                      # alpha0::Real,
                      # mayterminate::Bool;
                      c1::Real = 1e-4,
                      c2::Real = 0.9,
                      rho::Real = 2.0,
                      kwargs...)

    println("à l'entrée de strongwolfe2 #f=",df.nlp.counters.neval_obj," #g=",df.nlp.counters.neval_grad)
    # TODO: do we need gr_new anymore?
    # Any call to gradient! or value_gradient! would update df.g anyway
    # println("on est dans strongwolfe2")

    # Parameter space
    # n'est pas nécessaire, car on update x à l'extérieur du linesearch
    #n = length(x)

    # Step-sizes
    a_0 = 0.0
    a_iminus1 = a_0
    # a_i = alpha0
    a_i = 1.0 # il semble que alpha0 soit systématiquement 1.0...
    a_max = 65536.0

    # phi(alpha) = df.f(x + alpha * p)
    # phi_0 = lsr.value[end]
    # on utilise pas lsr... je ne comprend pas à quoi ça sert...
    # comme on a h₀ on a directement phi_0
    phi_0 = h₀
    phi_a_iminus1 = phi_0
    phi_a_i = NaN

    # phi'(alpha) = vecdot(g(x + alpha * p), p)
    # phiprime_0 = lsr.slope[end]
    # même principe on peut directement utilisé g₀
    phiprime_0 = g₀
    phiprime_a_i = NaN

    # Iteration counter
    i = 1

    # println("a_0=",a_0," a_iminus1=",a_iminus1," a_i=",a_i," a_max=",a_max)
    # println("phi_0=",phi_0," phi_a_iminus1=",phi_a_iminus1," phi_a_i=",phi_a_i)
    # println("phiprime_0=",phiprime_0," phiprime_a_i=",phiprime_a_i)

    # println("on a tout les paramètres de strongwolfe2")

    while a_i < a_max
        #println("on est dans le while de strongwolfe2")
        # Update x_new
        # On update x seulement à l'extérieur du linesearch
        # @simd for index in 1:n
        #     @inbounds x_new[index] = x[index] + a_i * p[index]
        # end

        # Evaluate phi(a_i)
        # phi_a_i = NLSolversBase.value!(df, x_new)
        phi_a_i = obj(df,a_i)

        #println("on a phi_a_i")

        # Test Wolfe conditions
        if (phi_a_i > phi_0 + c1 * a_i * phiprime_0) || ( (phi_a_i >= phi_a_iminus1) && (i > 1))
            # a_star = zoom(a_iminus1, a_i,
            #               phiprime_0, phi_0,
            #               df, x, p, x_new, gr_new)
            # on utilise le zoom qui existe déjà mais avec les paramètres que l'on a déjà
            #println("on est dans le premier if")
            #println("on rentre dans zoom2 1")
            #println("(a_iminus1,a_i,phiprime_0,phi_0)=",(a_iminus1,a_i,phi_0,phiprime_0))
            println("juste avant de rentré dans zomm2 #f=",df.nlp.counters.neval_obj," #g=",df.nlp.counters.neval_grad)
            a_star = zoom2(a_iminus1, a_i, phiprime_0, phi_0, df)#, gr_new)
            println("à la sortie de zoom2 #f=",df.nlp.counters.neval_obj," #g=",df.nlp.counters.neval_grad)
            return (a_star, true, obj(df,a_star), i, 0.0)
        end

        # Evaluate phi'(a_i)
        # NLSolversBase.gradient!(df, x_new)
        # gr_new[:] = gradient(df)
        #
        # phiprime_a_i = vecdot(gr_new, p)
        # on utilise les outils que l'on a déjà pour calculer  φ'(aᵢ)
        phiprime_a_i = grad(df,a_i)
        #println("on a phiprime_a_i")
        # Check condition 2
        if abs(phiprime_a_i) <= -c2 * phiprime_0
          println("quand on sort de strongwolfe2 sans passer par zoom #f=",df.nlp.counters.neval_obj," #g=",df.nlp.counters.neval_grad)
          return (a_i,true,obj(df,a_i),i, 0.0)
        end

        # Check condition 3
        if phiprime_a_i >= 0.0
            println("juste avant de rentré dans zomm2 #f=",df.nlp.counters.neval_obj," #g=",df.nlp.counters.neval_grad)
            a_star = zoom2(a_i, a_iminus1, phiprime_0, phi_0, df)#, gr_new)
            #println("on rentre dans zoom2 2")
            println("à la sortie de zoom2 #f=",df.nlp.counters.neval_obj," #g=",df.nlp.counters.neval_grad)
            return (a_star, true, obj(df,a_star), i, 0.0)
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
    #println("on sort de strongwolfe2")
    return (a_max, false, obj(df,a_max), i, 0.0)
end

function zoom2(a_lo::Real,
              a_hi::Real,
              phiprime_0::Real,
              phi_0::Real,
              df :: AbstractLineFunction;
              #p::Vector;
              #gr_new::Vector;
              c1::Real = 1e-4,
              c2::Real = 0.9)

    # println("on est dans zoom2")
    # Parameter space
    # pour les mêmes raisons que dans strongwolfe on utilise pas n
    # n = length(x)

    # Step-size
    a_j = NaN

    # Count iterations
    iteration = 0
    max_iterations = 10

    # Shrink bracket
    while iteration < max_iterations
        iteration += 1

        # Cache phi_a_lo
        # étant donné qu'on update pas x dans le linesearch
        # on a pas besoin de x_new
        # @simd for index in 1:n
        #     @inbounds x_new[index] = x[index] + a_lo * p[index]
        # end

        # phi_a_lo = NLSolversBase.value_gradient!(df, x_new)
        # gr_new[:] = NLSolversBase.gradient(df)
        # phiprime_a_lo = vecdot(gr_new, p)
        # on utilise nos outils pour calculer le gradient de a_lo
        phi_a_lo = obj(df,a_lo)
        phiprime_a_lo = grad(df, a_lo)


        # Cache phi_a_hi
        # étant donné qu'on update pas x dans le linesearch
        # on a pas besoin de x_new
        # @simd for index in 1:n
        #     @inbounds x_new[index] = x[index] + a_hi * p[index]
        # end

        # phi_a_hi = NLSolversBase.value_gradient!(df, x_new)
        # gr_new[:] = NLSolversBase.gradient(df)
        # phiprime_a_hi = vecdot(gr_new, p)
        # on utilise nos outils pour calculer le grandient de a_hi
        phi_a_hi = obj(df,a_hi)
        phiprime_a_hi = grad(df,a_hi)

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

        # Update x_new
        # on utilise pas x dans nos trucs
        # @simd for index in 1:n
        #     @inbounds x_new[index] = x[index] + a_j * p[index]
        # end

        # Evaluate phi(a_j)
        # phi_a_j = NLSolversBase.value!(df, x_new)
        # on le calcul avec nos outils
        phi_a_j = obj(df,a_j)

        # Check Armijo
        if (phi_a_j > phi_0 + c1 * a_j * phiprime_0) ||(phi_a_j > phi_a_lo)
            a_hi = a_j
        else
            # Evaluate phiprime(a_j)
            # NLSolversBase.gradient!(df, x_new)
            # gr_new[:] = gradient(df)
            # phiprime_a_j = vecdot(gr_new, p)
            # on utilise nos outils pour calculer le gradient de notre point courant
            phiprime_a_j = grad(df, a_j)

            if abs(phiprime_a_j) <= -c2 * phiprime_0
                # println("on sort de zoom2 i=",iteration)
                return a_j
            end

            if phiprime_a_j * (a_hi - a_lo) >= 0.0
                a_hi = a_lo
            end

            a_lo = a_j
        end
    end

    # Quasi-error response
    # println("on sort de zoom2 i=",iteration)
    return a_j
end

# a_lo = a_{i - 1}
# a_hi = a_{i}
function interpolate2(a_i1::Real, a_i::Real,
                     phi_a_i1::Real, phi_a_i::Real,
                     phiprime_a_i1::Real, phiprime_a_i::Real)
    d1 = phiprime_a_i1 + phiprime_a_i -
        3.0 * (phi_a_i1 - phi_a_i) / (a_i1 - a_i)
    #println("sign(d1 * d1 - phiprime_a_i1 * phiprime_a_i)=",sign(d1 * d1 - phiprime_a_i1 * phiprime_a_i))
    if (d1 * d1 - phiprime_a_i1 * phiprime_a_i)>0
      d2 = sqrt(d1 * d1 - phiprime_a_i1 * phiprime_a_i)
      return a_i - (a_i - a_i1) *
          ((phiprime_a_i + d2 - d1) /
          ( phiprime_a_i - phiprime_a_i1 + 2.0 * d2))
    else
      return a_i + (a_i - a_i1)/(phi_a_i - phi_a_i1)
    end
end
