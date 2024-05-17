# benchmark_GLM
Benchopt benchmark for Generalized Linear Models

Install
--------

For now, only Linear Regression models are optimized correctly, you can use commands like the following:

	$ benchopt run . -o GLM[model=linreg] -s NESTEROV-GD -s L-BFGS-B -d simulated[binary=false] -d diabetes


Use ``benchopt run -h`` for more details about the available options, or visit https://benchopt.github.io/api.html.
