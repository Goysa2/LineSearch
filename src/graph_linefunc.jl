export graph_linefunc
function graph_linefunc(h :: AbstractLineFunction2;
                        precision :: Float64 = 0.025,
                        a :: Float64 = 0.0,
                        b :: Float64 = 25.0,
                        color :: Symbol = :red,
                        verboseGraph :: Bool = false,
                        kwargs...)

  x_axis = []
  y_axis = []
  i = a

  while i <= b
    push!(x_axis, i)
    push!(y_axis, obj(h,i))
    i += precision
  end

  verboseGraph && println(" ")
  verboseGraph && println("x_axis = $x_axis")
  verboseGraph && println("y_axis = $y_axis")

  graph_line_func = scatter(x_axis, y_axis, color = color)

  display(graph_line_func)

  return graph_line_func

end
