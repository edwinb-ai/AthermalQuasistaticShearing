using LinearAlgebra
using Random
using StaticArrays
using DelimitedFiles
using Printf: @sprintf

# For convenience, define a 2D mutable vector alias.
const Vec2 = MVector{2,Float64}

###############################
# Simulation Parameters Struct#
###############################
mutable struct SimulationParams
    Lx::Float64                    # Box length in x
    Ly::Float64                    # Box length in y
    N::Int                         # Number of particles
    r_cut::Float64                 # Interaction cutoff (in units of effective diameter)
    dgamma::Float64                # Strain increment per shear step
    lbfgs_tol::Float64             # L-BFGS convergence tolerance
    lbfgs_max_steps::Int           # Maximum L-BFGS iterations per shear step
    lbfgs_m::Int                   # L-BFGS memory parameter (number of stored vectors)
    plastic_threshold::Float64     # Threshold for (ΔE/Δγ) to detect a plastic event
    non_additivity::Float64        # Non-additivity parameter for the effective diameter
end

# Default parameters.
function default_params()
    return SimulationParams(
        10.0,      # Lx (will be overwritten if configuration file is used)
        10.0,      # Ly
        100,       # N (will be overwritten if configuration file is used)
        1.25,      # r_cut
        1e-4,      # dgamma (strain increment)
        1e-6,      # lbfgs_tol (L-BFGS convergence tolerance)
        100000,    # lbfgs_max_steps (L-BFGS max iterations)
        10,        # lbfgs_m (L-BFGS memory parameter)
        -1e-6,     # plastic_threshold (plastic event if ΔE/Δγ < threshold)
        0.2,       # non_additivity
    )
end

#############################################
# Read Configuration from File              #
#############################################
# The file is expected to have the following format:
#
#   <N_particles>
#   Lattice="Lx 0 0 0 Ly 0 0 0 Lz" Properties=...
#   <species> <id> <radius> <x> <y>
#   ...
#
# (For a 2D simulation only x and y are used.)
function read_configuration(filename::String)
    open(filename, "r") do io
        # First line: number of particles.
        line = readline(io)
        N_particles = parse(Int, strip(line))

        # Second line: header. Extract lattice information.
        header = readline(io)
        m = match(r"Lattice=\"([^\"]+)\"", header)
        if m === nothing
            error("Lattice information not found in header!")
        end
        lattice_str = m.captures[1]
        lattice_tokens = split(lattice_str)
        if length(lattice_tokens) < 9
            error("Unexpected lattice format!")
        end
        # For 2D, assume Lx is token 1 and Ly is token 5.
        Lx_file = parse(Float64, lattice_tokens[1])
        Ly_file = parse(Float64, lattice_tokens[5])

        # Initialize arrays.
        positions = Vector{Vec2}(undef, N_particles)
        diameters = Vector{Float64}(undef, N_particles)

        # Read particle data.
        for i in 1:N_particles
            line = readline(io)
            tokens = split(strip(line))
            if length(tokens) < 5
                error("Not enough data on line $i of particle data!")
            end
            # The file gives radii; convert to diameter.
            diameters[i] = parse(Float64, tokens[3]) * 2.0
            # Load the positions
            x = parse(Float64, tokens[4])
            y = parse(Float64, tokens[5])
            positions[i] = Vec2(x, y)
        end
        return positions, Lx_file, Ly_file, diameters
    end
end

#################################
# Utility: Periodic Wrapping    #
#################################
@inline function apply_periodic!(positions::Vector{Vec2}, γ::Float64, p::SimulationParams)
    @inbounds for pos in positions
        n_y = floor(Int, pos[2] / p.Ly)
        pos[2] -= n_y * p.Ly
        pos[1] -= n_y * γ * p.Ly
        n_x = floor(Int, pos[1] / p.Lx)
        pos[1] -= n_x * p.Lx
    end
end

############################################################
# Lees–Edwards Minimum Image Convention for Sheared Systems  #
############################################################
@inline function minimum_image(pi::Vec2, pj::Vec2, γ::Float64, p::SimulationParams)
    dx = pi[1] - pj[1]
    dy = pi[2] - pj[2]
    # safer integer for dy: round ≈ floor(x+0.5)
    n_y = floor(Int, dy / p.Ly + 0.5)
    dy -= n_y * p.Ly
    dx -= γ * p.Ly * n_y
    dx -= p.Lx * floor(dx / p.Lx + 0.5)
    return Vec2(dx, dy)
end

##########################################
# Pairwise Potential and Force Functions #
##########################################
function pair_potential_energy(r::Float64, σ_eff::Float64, params::SimulationParams)
    if r < params.r_cut * σ_eff
        term_1 = (σ_eff / r)^12
        term_2 = params.c2 * (r / σ_eff)^2
        term_3 = params.c4 * (r / σ_eff)^4
        return term_1 + params.c0 + term_2 + term_3
    else
        return 0.0
    end
end

function pair_force(r_vec::Vec2, r::Float64, σ_eff::Float64, params::SimulationParams)
    if r < params.r_cut * σ_eff
        force_mag =
            12.0 * σ_eff^12 / r^13 - 2.0 * params.c2 * r / (σ_eff^2) -
            4.0 * params.c4 * r^3 / (σ_eff^4)
        return (force_mag * r_vec) / r
    else
        return Vec2(0.0, 0.0)
    end
end

###############################
# Cell List Construction      #
###############################
function build_cell_list(positions::Vector{Vec2}, p::SimulationParams, d_max::Float64)
    max_r_dist = p.r_cut * d_max
    n_cells_x = max(Int(floor(p.Lx / max_r_dist)), 1)
    n_cells_y = max(Int(floor(p.Ly / max_r_dist)), 1)
    cell_size_x = p.Lx / n_cells_x
    cell_size_y = p.Ly / n_cells_y
    cell_list = [Int[] for _ in 1:n_cells_x, _ in 1:n_cells_y]
    @inbounds for (i, pos) in enumerate(positions)
        cx = mod(floor(Int, pos[1] / cell_size_x), n_cells_x) + 1
        cy = mod(floor(Int, pos[2] / cell_size_y), n_cells_y) + 1
        push!(cell_list[cx, cy], i)
    end
    return cell_list, n_cells_x, n_cells_y, cell_size_x, cell_size_y
end

# extra cells required so the neighbour stencil spans |Δx|=γ·Lx
@inline n_extra_cells(γ::Float64, cell_size_x::Float64, Ly::Float64) =
    ceil(Int, abs(γ) * Ly / cell_size_x)

##########################################
# Compute Forces and Total Energy        #
##########################################
function compute_forces(
    positions::Vector{Vec2}, diameters::Vector{Float64}, γ::Float64, p::SimulationParams
)
    Np = length(positions)
    forces = [Vec2(0.0, 0.0) for _ in 1:Np]
    energy = 0.0

    cell_list, nx, ny, cell_size_x, _ = build_cell_list(positions, p, maximum(diameters))
    extra = n_extra_cells(γ, cell_size_x, p.Ly)

    @inbounds for cx in 1:nx, cy in 1:ny
        cell_particles = cell_list[cx, cy]
        for i_local in eachindex(cell_particles)
            i = cell_particles[i_local]
            for dx in (-1 - extra):(1 + extra), dy in -1:1
                ncx = mod(cx - 1 + dx, nx) + 1
                ncy = mod(cy - 1 + dy, ny) + 1
                for j in cell_list[ncx, ncy]
                    (ncx == cx && ncy == cy && j <= i) && continue
                    disp = minimum_image(positions[i], positions[j], γ, p)
                    r = norm(disp)
                    σ_i, σ_j = diameters[i], diameters[j]
                    σ_eff = 0.5 * (σ_i + σ_j) * (1.0 - p.non_additivity * abs(σ_i - σ_j))
                    if r < p.r_cut * σ_eff
                        energy += pair_potential_energy(r, σ_eff, p)  # unchanged call
                        fpair = pair_force(disp, r, σ_eff, p)
                        forces[i] += fpair
                        forces[j] -= fpair
                    end
                end
            end
        end
    end
    return forces, energy
end

##########################################
# Compute Stress Tensor                  #
##########################################
function compute_stress_tensor(
    positions::Vector{Vec2}, diameters::Vector{Float64}, γ::Float64, p::SimulationParams
)
    V = p.Lx * p.Ly
    stress = zeros(2, 2)

    cell_list, nx, ny, cell_size_x, _ = build_cell_list(positions, p, maximum(diameters))
    extra = n_extra_cells(γ, cell_size_x, p.Ly)

    @inbounds for cx in 1:nx, cy in 1:ny
        cell_particles = cell_list[cx, cy]
        for i_local in eachindex(cell_particles)
            i = cell_particles[i_local]
            for dx in (-1 - extra):(1 + extra), dy in -1:1
                ncx = mod(cx - 1 + dx, nx) + 1
                ncy = mod(cy - 1 + dy, ny) + 1
                for j in cell_list[ncx, ncy]
                    (ncx == cx && ncy == cy && j <= i) && continue
                    disp = minimum_image(positions[i], positions[j], γ, p)
                    r = norm(disp)
                    σ_i, σ_j = diameters[i], diameters[j]
                    σ_eff = 0.5 * (σ_i + σ_j) * (1.0 - p.non_additivity * abs(σ_i - σ_j))
                    (r < p.r_cut * σ_eff) || continue
                    fpair = pair_force(disp, r, σ_eff, p)
                    # Stress is negative of the force times the displacement
                    stress .-= disp * transpose(fpair)
                end
            end
        end
    end
    return stress ./ V
end

##########################################
# Plastic Event Detection                #
##########################################
function plastic_event_detected(
    e_prev::Float64, e_current::Float64, dgamma::Float64, threshold::Float64
)
    dE_dgamma = (e_current - e_prev) / dgamma
    return dE_dgamma < threshold
end

##########################################################################
# Helper: Evaluate candidate point along the search direction
##########################################################################
function evaluate_candidate(
    α::Float64,
    x_old::Vector{Vec2},
    d::Vector{Vec2},
    diameters::Vector{Float64},
    gamma::Float64,
    params::SimulationParams,
)
    # Compute candidate positions: x_candidate = x_old + α * d
    candidate = [x_old[i] + α * d[i] for i in 1:length(x_old)]
    apply_periodic!(candidate, gamma, params)
    forces, E = compute_forces(candidate, diameters, gamma, params)
    # Gradient is minus the forces.
    candidate_grad = [-f for f in forces]
    return candidate, E, candidate_grad
end

##########################################################################
# Helper: Zoom procedure to find an acceptable α in [α_lo, α_hi]
##########################################################################
function zoom(
    α_lo::Float64,
    α_hi::Float64,
    x_old::Vector{Vec2},
    d::Vector{Vec2},
    E_current::Float64,
    d_dot_g_current::Float64,
    diameters::Vector{Float64},
    gamma::Float64,
    params::SimulationParams,
    c1::Float64,
    c2::Float64,
)
    max_zoom_iter = 20
    candidate = Vector{Vec2}()
    E_candidate = Inf
    candidate_grad = Vector{Vec2}()
    α_j = 0.0
    for iter in 1:max_zoom_iter
        α_j = (α_lo + α_hi) / 2.0
        candidate, E_candidate, candidate_grad = evaluate_candidate(
            α_j, x_old, d, diameters, gamma, params
        )
        # Evaluate Armijo condition at α_j.
        if (E_candidate > E_current + c1 * α_j * d_dot_g_current) ||
            (E_candidate >= evaluate_candidate(α_lo, x_old, d, diameters, gamma, params)[2])
            α_hi = α_j
        else
            d_dot_g_candidate = sum(dot(d[i], candidate_grad[i]) for i in 1:length(d))
            if abs(d_dot_g_candidate) <= -c2 * d_dot_g_current
                return α_j, candidate, E_candidate, candidate_grad
            end
            if d_dot_g_candidate * (α_hi - α_lo) >= 0
                α_hi = α_lo
            end
            α_lo = α_j
        end
    end
    return α_j, candidate, E_candidate, candidate_grad
end

##########################################################################
# Wolfe Conditions Based Line Search (Strong Wolfe)
##########################################################################
function line_search_wolfe!(
    x_old::Vector{Vec2},
    diameters::Vector{Float64},
    gamma::Float64,
    params::SimulationParams,
    d::Vector{Vec2},
    E_current::Float64,
    g_current::Vector{Vec2},
    c1::Float64,
    c2::Float64,
)
    # Compute directional derivative at the starting point:
    d_dot_g_current = sum(dot(d[i], g_current[i]) for i in 1:length(d))
    α_prev = 0.0
    α = 1.0  # initial trial step
    candidate, E_candidate, candidate_grad = evaluate_candidate(
        α, x_old, d, diameters, gamma, params
    )
    max_iter = 20
    for iter in 1:max_iter
        if (E_candidate > E_current + c1 * α * d_dot_g_current) || (
            iter > 1 &&
            E_candidate >=
            evaluate_candidate(α_prev, x_old, d, diameters, gamma, params)[2]
        )
            # If not, zoom between α_prev and α.
            return zoom(
                α_prev,
                α,
                x_old,
                d,
                E_current,
                d_dot_g_current,
                diameters,
                gamma,
                params,
                c1,
                c2,
            )
        end
        d_dot_g_candidate = sum(dot(d[i], candidate_grad[i]) for i in 1:length(d))
        if abs(d_dot_g_candidate) <= -c2 * d_dot_g_current
            return α, candidate, E_candidate, candidate_grad
        end
        if d_dot_g_candidate >= 0
            return zoom(
                α,
                α_prev,
                x_old,
                d,
                E_current,
                d_dot_g_current,
                diameters,
                gamma,
                params,
                c1,
                c2,
            )
        end
        α_prev = α
        α *= 2.0  # increase step size
        candidate, E_candidate, candidate_grad = evaluate_candidate(
            α, x_old, d, diameters, gamma, params
        )
    end
    return α, candidate, E_candidate, candidate_grad
end

##########################################################################
# L-BFGS Energy Minimization
##########################################################################
function lbfgs_minimization!(
    positions::Vector{Vec2},
    diameters::Vector{Float64},
    gamma::Float64,
    params::SimulationParams,
)
    Np = length(positions)
    m = params.lbfgs_m  # Memory parameter

    # Compute initial forces and energy.
    forces, energy = compute_forces(positions, diameters, gamma, params)
    # Gradient: g = -forces.
    g = [-f for f in forces]

    # Initialize L-BFGS storage
    s_history = Vector{Vector{Vec2}}()  # Position differences s_k = x_{k+1} - x_k
    y_history = Vector{Vector{Vec2}}()  # Gradient differences y_k = g_{k+1} - g_k
    rho_history = Vector{Float64}()     # 1 / (y_k^T s_k)

    no_progress_limit = 50
    no_progress_counter = 0
    best_gradient_norm = Inf
    convergence = false

    # Strong Wolfe parameters
    c1 = 1e-4
    c2 = 0.9

    # Save current positions and gradient
    x_old = [copy(positions[i]) for i in 1:Np]
    g_old = [copy(g[i]) for i in 1:Np]

    for iter in 1:(params.lbfgs_max_steps)
        # Check convergence: norm of gradient (force).
        gradient_norm = sqrt(sum(norm(gi)^2 for gi in g))

        if gradient_norm < params.lbfgs_tol
            convergence = true
            return energy, gradient_norm, convergence
        end

        # Track progress
        if gradient_norm < best_gradient_norm * 0.99
            best_gradient_norm = gradient_norm
            no_progress_counter = 0
        else
            no_progress_counter += 1
        end

        # Reset L-BFGS if no progress
        if no_progress_counter >= no_progress_limit
            empty!(s_history)
            empty!(y_history)
            empty!(rho_history)
            no_progress_counter = 0
        end

        # Compute search direction using L-BFGS two-loop recursion
        d = lbfgs_two_loop_recursion(g, s_history, y_history, rho_history)

        # Check if search direction is a descent direction
        d_dot_g = sum(dot(d[i], g[i]) for i in 1:Np)
        if d_dot_g >= 0
            # Reset to steepest descent
            d = [-g_i for g_i in g]
            # Clear L-BFGS history
            empty!(s_history)
            empty!(y_history)
            empty!(rho_history)
            d_dot_g = sum(dot(d[i], g[i]) for i in 1:Np)
        end

        # Perform line search using strong Wolfe conditions
        α, candidate, E_candidate, candidate_grad = line_search_wolfe!(
            x_old, diameters, gamma, params, d, energy, g, c1, c2
        )

        # Compute s_k = x_{k+1} - x_k
        s_k = [candidate[i] - x_old[i] for i in 1:Np]

        # Compute y_k = g_{k+1} - g_k
        y_k = [candidate_grad[i] - g[i] for i in 1:Np]

        # Compute rho_k = 1 / (y_k^T s_k)
        y_dot_s = sum(dot(y_k[i], s_k[i]) for i in 1:Np)

        # Only update L-BFGS vectors if curvature condition is satisfied
        if y_dot_s > 1e-14
            rho_k = 1.0 / y_dot_s

            # Add to history
            push!(s_history, s_k)
            push!(y_history, y_k)
            push!(rho_history, rho_k)

            # Maintain limited memory
            if length(s_history) > m
                popfirst!(s_history)
                popfirst!(y_history)
                popfirst!(rho_history)
            end
        end

        # Update positions, energy, and gradient
        for i in 1:Np
            positions[i] = candidate[i]
        end
        energy = E_candidate

        # Update for next iteration
        x_old = [copy(positions[i]) for i in 1:Np]
        g = candidate_grad
    end

    forces, energy = compute_forces(positions, diameters, gamma, params)
    gradient_norm = sqrt(sum(norm(f)^2 for f in forces))

    @warn "L-BFGS did not converge after $(params.lbfgs_max_steps) steps; final gradient norm = $(gradient_norm)"

    return energy, gradient_norm, convergence
end

##########################################################################
# L-BFGS Two-Loop Recursion for Computing Search Direction
##########################################################################
function lbfgs_two_loop_recursion(
    g::Vector{Vec2},
    s_history::Vector{Vector{Vec2}},
    y_history::Vector{Vector{Vec2}},
    rho_history::Vector{Float64},
)
    # If no history, return steepest descent
    if isempty(s_history)
        return [-g_i for g_i in g]
    end

    k = length(s_history)
    alpha = zeros(k)

    # Initialize q with current gradient
    q = [copy(g[i]) for i in 1:length(g)]

    # First loop (backwards)
    for i in k:-1:1
        alpha[i] = rho_history[i] * sum(dot(s_history[i][j], q[j]) for j in eachindex(q))
        for j in eachindex(q)
            q[j] -= alpha[i] * y_history[i][j]
        end
    end

    # Initial Hessian approximation (scaling)
    if k > 0
        # Use γ_k = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1}) as initial scaling
        s_dot_y = sum(
            dot(s_history[end][i], y_history[end][i]) for i in 1:length(s_history[end])
        )
        y_dot_y = sum(
            dot(y_history[end][i], y_history[end][i]) for i in 1:length(y_history[end])
        )
        if y_dot_y > 1e-14
            gamma = s_dot_y / y_dot_y
        else
            gamma = 1.0
        end

        # Apply initial Hessian: r = H_0 * q = γ * q
        r = [gamma * q[i] for i in eachindex(q)]
    else
        r = [copy(q[i]) for i in eachindex(q)]
    end

    # Second loop (forwards)
    for i in 1:k
        beta = rho_history[i] * sum(dot(y_history[i][j], r[j]) for j in eachindex(r))
        for j in eachindex(r)
            r[j] += (alpha[i] - beta) * s_history[i][j]
        end
    end

    # Return negative of r (search direction)
    return [-r[i] for i in eachindex(r)]
end

##############################################
# Configuration Saving Function
##############################################
function save_configuration(
    filename::String,
    positions::Vector{Vec2},
    diameters::Vector{Float64},
    params::SimulationParams,
)
    open(filename, "w") do f
        println(f, length(positions))
        println(
            f,
            "Lattice=\"$(params.Lx) 0.0 0.0 0.0 $(params.Ly) 0.0 0.0 0.0 0.0\" Properties=type:I:1:id:I:1:radius:R:1:pos:R:2",
        )
        for i in 1:length(positions)
            x = positions[i][1]
            y = positions[i][2]
            radius = diameters[i] / 2.0
            println(f, "1 $i $radius $x $y")
        end
    end
end

##############################################
# Main Simulation: Athermal Quasistatic Shear #
##############################################
function run_athermal_quasistatic(filename::Union{Nothing,String}=nothing)
    params = default_params()
    positions = Vector{Vec2}()
    diameters = Vector{Float64}()
    if filename !== nothing
        positions_file, Lx_file, Ly_file, diameters_file = read_configuration(filename)
        positions = positions_file
        diameters = diameters_file
        params.Lx = Lx_file
        params.Ly = Ly_file
        params.N = length(positions)
        println("Configuration loaded from file:")
        println("  Number of particles: $(params.N)")
        println("  Lx = $(params.Lx), Ly = $(params.Ly)")
    else
        params.N = params.N
        positions = [Vec2(rand() * params.Lx, rand() * params.Ly) for _ in 1:(params.N)]
        diameters = ones(Float64, params.N)
    end

    # Define the parameters for shearing
    params.dgamma = 1e-4
    gamma_max = 0.2
    gamma = 0.0
    # Initial energy minimization.
    println("Performing initial energy minimization (γ = $gamma)...")
    (e_prev, grad_norm, convergence) = lbfgs_minimization!(
        positions, diameters, gamma, params
    )
    # Check if L-BFGS converged
    if !convergence
        @error "Initial energy minimization did not converge!"
        return nothing
    end
    # Normalize the energy per particle.
    e_prev /= params.N
    println("γ = $gamma, Energy per particle = $e_prev, Gradient norm = $grad_norm")
    println("Initial Stress tensor:")
    println(compute_stress_tensor(positions, diameters, gamma, params))

    # Create a new directory for saving all results
    save_dir = mkdir("lbfgs_results")
    # Save the initial configuration.
    save_configuration(
        joinpath(save_dir, "initial_configuration.xyz"), positions, diameters, params
    )

    # Let's open a file to save the energy information at every step
    energy_file = open(joinpath(save_dir, "energy_aqs_lbfgs.txt"), "w")
    stress_file = open(joinpath(save_dir, "stress_aqs_lbfgs.txt"), "w")

    step = 0
    # Main loop: apply shear until a plastic event is detected.
    while gamma < gamma_max
        step += 1
        # Apply affine shear: x' = x + dγ * y.
        for pos in positions
            pos[1] += params.dgamma * pos[2]
        end
        gamma += params.dgamma
        apply_periodic!(positions, gamma, params)
        (e_current, grad_norm, convergence) = lbfgs_minimization!(
            positions, diameters, gamma, params
        )
        # Normalize the energy per particle.
        e_current /= params.N

        # Write and flush the file
        println(energy_file, e_current)
        flush(energy_file)

        println(
            "Step $step: γ = $gamma, Energy per particle = $e_current, Gradient norm = $grad_norm",
        )
        # Write the xy component of the stress tensor to file
        stress_value = compute_stress_tensor(positions, diameters, gamma, params)
        writedlm(stress_file, [gamma stress_value[1, 2]])
        flush(stress_file)

        # if plastic_event_detected(
        #     e_prev, e_current, params.dgamma, params.plastic_threshold
        # )
        #     println("Plastic event detected at γ = $gamma (step $step)!")
        #     # println("Reversing strain direction.")
        #     # params.dgamma = -params.dgamma
        #     # Optionally, save the configuration at this reversal.
        #     save_configuration("plastic_event_γ=$(gamma).xyz", positions, diameters, params)
        #     break
        # end
        e_prev = e_current

        savename_configuration = joinpath(save_dir, @sprintf("conf_%.4g.xyz", gamma))
        save_configuration(savename_configuration, positions, diameters, params)
    end

    # (Optional) At the end, save the final configuration.
    # save_configuration("final_configuration.xyz", positions, diameters, params)
    close(energy_file)
    close(stress_file)

    return nothing
end

###########################
# Run the Simulation      #
###########################
run_athermal_quasistatic("init.xyz")
