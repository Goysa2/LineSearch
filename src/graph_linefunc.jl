export graph_linefunc

#tool designed to visualize the Armijo and Wolfe condition
#when given a C1LineFunction or a C2LineFunction
#Useful for debugging because it can help to visualize in a particular iteration
#But costs a lot of function evaluation so might need improvement

function graph_linefunc(h :: AbstractLineFunction2,
                        h₀ :: Float64,
                        g₀ :: Float64;
                        τ₀ :: Float64 = 1e-4,
                        τ₁ :: Float64 = 0.9999,
                        precision :: Float64 = 0.025,
                        a :: Float64 = 0.0,
                        b :: Float64 = 25.0,
                        color :: Symbol = :red,
                        verboseGraph :: Bool = false,
                        kwargs...)

  PyPlot.clf()               #clear the existing figure...

  x_axis = []
  y_axis = []
  i = a

  while i <= b               #"shape" of h
    push!(x_axis, i)
    push!(y_axis, obj(h,i))
    i += precision
  end

  x = PyPlot.linspace(a, b,200)
  y = (τ₀*g₀)*x + h₀                       #armijo condition

  PyPlot.figure(1)
  PyPlot.plot(x,y)                                      #we "put" the armijo condition in the graph
  PyPlot.scatter(x_axis,y_axis, color = color, s = 10.0) #we put the "shape" of h in the graph



end
