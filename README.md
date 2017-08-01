# LineSearch

[![Build Status](https://travis-ci.org/Goysa2/LineSearch.jl.svg?branch=master)](https://travis-ci.org/Goysa2/LineSearch.jl)

[![Coverage Status](https://coveralls.io/repos/Goysa2/LineSearch.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/Goysa2/LineSearch.jl?branch=master)

[![codecov.io](http://codecov.io/github/Goysa2/LineSearch.jl/coverage.svg?branch=master)](http://codecov.io/github/Goysa2/LineSearch.jl?branch=master)

## Installing
`julia> Pkg.clone("https://github.com/Goysa2/LineSearch.git")`
`julia> Pkg.build("LineSearch")`

## How to use
This package is a collection of LineSearch algorithms made to be used with the
descent algorithms presented in the
[LSDescentMethods](https://github.com/vepiteski/LSDescentMethods) package.

The hyper parameters common to all line searches are presented in the abstract_
linesearch. Specific key words are presented in the functions themselves.
s
## Other Line Search
The algorithms presented in Other-LS are interfaced from the
[LineSearches package](https://github.com/JuliaNLSolvers/LineSearches.jl). They
consist of the Hager & Zhang line search and the More & Thuente line searhch.
There is also an Armijo backtracking process and the Nocedal & Wright line
search, but those are provided here as well so the "Other line searches" are
mostly used for Hager & Zhang and More & Thuente line search.


## References
J. Nocedal and S.Wright
[Numerical Optimization](http://www.bioinfo.org.cn/~wangchao/maa/Numerical_Optimization.pdf)

J.P Dusseault Univariate diffentiable optimization algorithms and LineSearch
computation
