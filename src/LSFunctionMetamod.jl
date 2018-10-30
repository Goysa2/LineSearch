export FunctionMeta, LS_Function_Meta

abstract type FunctionMeta end

mutable struct LS_Function_Meta

    # Successful and very succesful iteration threshold for ARC/TR variant
    eps1 :: Float64
    eps2 :: Float64

    # Reduction/Augmentation size of the trust region/cubic regularisation paramter
    # for the ARC/TR varitans
    aug :: Float64
    red :: Float64

    # Trust region/Cubic regularisation parameter for the ARC/TR variants
    Δ  :: Float64
    Δn :: Float64  # only for the TR variants
    Δp :: Float64  # only for the TR variants

    # Direction used (Newton's, Secant or Improved Secant)
    dir :: String

    # Additionnal step. If true, this option will make another line search iteration
    # after finding an admissible step size.
    add_step :: Bool ## Pertinent à garder ?? Sam

        function LS_Function_Meta(;eps1 :: Float64 = 0.25,
                                  eps2  :: Float64 = 0.75,
                                  aug   :: Float64 = 0.7,
                                  red   :: Float64 = 0.1,
                                  Δ     :: Float64 = 10.0,
                                  dir   :: String = "Nwt",
                                  add_step :: Bool = false)

            Δp = Δ
            Δn = max(0.0, -Δ)

            return new(eps1, eps2, aug, red, Δ, Δn, Δp, dir, add_step)
        end # function
end # struct
