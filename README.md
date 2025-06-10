# Athermal Quasistatic Shearing

This repository holds codes to perform the athermal quasistatic shearing
(AQS) protocol. The protocol follows quasistatic dynamics, where a small
shear is applied on the side of the simulation box, and the configuration
is then minimized, to always maintain mechanical equilibrium.

# System

The system of choice is an amorphous solid that has glassy-like dynamics
and characteristics, taken originally from the work in [1]. The system
is a **polydisperse** mixture of point-like particles enclosed in a
square simulation box, making the simulation two-dimensional.

# Simulation method

To equilibrate the system, the `swapmc.jl` code enables the fast equilibration
of the system by using the swap Monte Carlo technique, where the diameter
is swapped between particles. This is a non-physical rule, but it makes it
easier to equilibrate the system. One can choose the number of particles,
the reduced density and the reduced temperature.

# AQS simulation method

The implementation of the AQS protocol is very standard. At every step,
the system is displaced by a small amount, and Lees-Edwards boundary conditions
are enforced, simulating a Couette flow. After displacing the particles, the
system is minimized to its closest energy minimum using the fast inertial
relaxation engine (FIRE) algorithm. This is the implementation in `aqs.jl`.

Alternatively, `aqs_cg.jl` implements a Polak-Ribiere conjugate gradient
minimization algorithm, using a line search that satisfies the Wolfe
conditions. However, the FIRE algorithm is preferred since it can reach
minima in less function calls than conjugate gradient.

Other versions have been included while checking for correctness of the codes and the implementation. For instance, the `aqs_lbfgs.jl` implements the L-BFGS optimization algorithm. The file `aqs_cgopt.jl` tries to use Julia packages to compute the cell lists and the optimization, but this code does not work. The code `aqs_cg_back.jl` implements the same Polak-Ribiere conjugate gradient minimization algorithm, but with backtracking instead of using the strong Wolfe conditions. Finally, `aqs_simple.jl` use the FIRE algorithm but does not use cell lists for computing the forces and the stress tensor, it uses a brute-force double-loop approach.

# References

1. Ninarello, A. Models and Algorithms for the Next Generation of Glass Transition Studies. Phys. Rev. X 7, (2017).
