import NLPModels: obj, grad, grad!, hess, objgrad

export LineModel
export obj, grad, derivative, grad!, derivative!, hess, redirect!, objgrad

"""A type to represent the restriction of a function to a direction.
Given f : R → Rⁿ, x ∈ Rⁿ and a nonzero direction d ∈ Rⁿ,

    ϕ = LineModel(nlp, x, d)

represents the function ϕ : R → R defined by

    ϕ(t) := f(x + td).
"""
mutable struct LineModel <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
  nlp :: AbstractNLPModel
  x :: Vector
  d :: Vector
  ∇ft :: Vector
  ## obj_grad :: Any ### On a dit qu'on voulait le obj_grad dans LineModel
  ## mais qu'est-ce qu'on veut? une méthode ? un compteur?
  ## la méthode obj_grad existe déjà et modifie les compteur de obj et grad
end

function LineModel(nlp :: AbstractNLPModel,
                   x   :: Vector{Float64},
                   d   :: Vector{Float64})
  meta = NLPModelMeta(1, x0=zeros(1), name="LineModel to $(nlp.meta.name))")
  g = Vector{Float64}(undef, size(x)[1])
  return LineModel(meta, Counters(), nlp, x, d, g)
end

"""`redirect!(ϕ, x, d)`

Change the values of x and d of the LineModel ϕ, but retains the counters.
"""
function redirect!(ϕ :: LineModel, x :: Vector{Float64}, d :: Vector{Float64})
  ϕ.x, ϕ.d = x, d
  return ϕ
end

"""`obj(f, t)` evaluates the objective of the `LineModel`

    ϕ(t) := f(x + td).
"""
function obj(f :: LineModel, t :: Float64)
  NLPModels.increment!(f, :neval_obj)
  return obj(f.nlp, f.x + t * f.d)
end

"""`grad(f, t)` evaluates the first derivative of the `LineModel`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.
"""
function grad(f :: LineModel, t :: Float64)
  NLPModels.increment!(f, :neval_grad)
  f.∇ft = grad(f.nlp, f.x + t * f.d)
  return dot(f.∇ft, f.d)
end
derivative(f :: LineModel, t :: Float64) = grad(f, t)

"""`grad!(f, t, g)` evaluates the first derivative of the `LineModel`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.

The gradient ∇f(x + td) is stored in `g`.
"""
function grad!(f :: LineModel, t :: Float64, g :: Vector{Float64})
  NLPModels.increment!(f, :neval_grad)
  f.∇ft = grad(f.nlp, f.x + t * f.d)
  g = dot(f.∇ft, f.d)
  return g
end
derivative!(f :: LineModel, t :: Float64, g :: Vector{Float64}) = grad!(f, t, g)

"""Evaluate the second derivative of the `LineModel`

    ϕ(t) := f(x + td),

i.e.,

    ϕ"(t) = dᵀ∇²f(x + td)d.
"""
function hess(f :: LineModel, t :: Float64)
  NLPModels.increment!(f, :neval_hess)
  return dot(f.d, hprod(f.nlp, f.x + t * f.d, f.d))
end

"""Evaluate the objective function and the derivative at the same time"""
function objgrad(f :: LineModel, t :: Float64)
	NLPModels.increment!(f, :neval_obj); NLPModels.increment!(f, :neval_grad)
  	f.∇ft = grad(f.nlp, f.x + t * f.d)
	obj_f = obj(f.nlp, f.x + t * f.d)
	return obj_f, dot(f.∇ft, f.d)
end

#end of module
# end
