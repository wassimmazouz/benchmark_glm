# benchmark_GLM
Benchopt benchmark for Generalized Linear Models

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/wassimmazouz/benchmark_glm.git
   $ cd benchmark_glm
   $ benchopt run .

For now, only Linear Regression models are optimized correctly, you can use commands like the following:

.. code-block::

	$ benchopt run . -o GLM[model=linreg] -s NESTEROV-GD -s L-BFGS-B -d simulated[binary=false] -d diabetes


Use ``benchopt run -h`` for more details about the available options, or visit https://benchopt.github.io/api.html.
