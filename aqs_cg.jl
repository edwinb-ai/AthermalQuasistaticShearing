using LinearAlgebra
using Random
using StaticArrays

# For convenience, define a 2D mutable vector alias.
const Vec2 = MVector{2,Float64}

###############################
# Simulation Parameters Struct#
###############################
mutable struct SimulationParams
    Lx::Float64           # Box length in x
    Ly::Float64           # Box length in y
    N::Int                # Number of particles
    r_cut::Float64        # Interaction cutoff
    dt_initial::Float64   # Initial timestep for FIRE
    dt_max::Float64       # Maximum allowed timestep for FIRE
    f_inc::Float64        # FIRE dt increase factor
    f_dec::Float64        # FIRE dt decrease factor
    alpha0::Float64       # FIRE initial mixing parameter
    dgamma::Float64       # Strain increment per shear step
    fire_tol::Float64     # FIRE convergence tolerance
    fire_max_steps::Int   # Maximum FIRE iterations per shear step
    plastic_threshold::Float64  # Threshold for (ΔE/Δγ) to detect a plastic event
    non_additivity::Float64     # Non-additivity parameter for the effective diameter
end

# Default parameters.
function default_params()
    return SimulationParams(
        10.0,      # Lx (will be overwritten if configuration file is used)
        10.0,      # Ly
        100,       # N (will be overwritten if configuration file is used)
        1.25,       # r_cut
        1e-3,      # dt_initial
        1e-2,      # dt_max
        0.9,       # f_inc
        0.4,       # f_dec
        0.1,       # alpha0
        1e-5,      # dgamma (strain increment)
        1e-4,      # fire_tol
        100000,     # fire_max_steps
        0.0,       # plastic_threshold (plastic event if ΔE/Δγ < threshold)
        0.2        # non_additivity
    )
end

#############################################
# Read Configuration from File              #
#############################################
# The file is expected to have the following format:
#
#   <N_particles>
#   Lattice="Lx 0 0 Lx" Properties=...
#   <species> <x> <y> <radius>
#   ...
#
# For a 2D simulation only x and y are used.
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
        if length(lattice_tokens) > 4
            error("Unexpected lattice format!")
        end
        # For 2D, assume Lx is token 1 and Ly is token 4.
        Lx_file = parse(Float64, lattice_tokens[1])
        Ly_file = parse(Float64, lattice_tokens[4])

        # Initialize arrays.
        positions = Vector{Vec2}(undef, N_particles)
        diameters = Vector{Float64}(undef, N_particles)

        # Read particle data.
        for i in 1:N_particles
            line = readline(io)
            tokens = split(strip(line))
            if length(tokens) < 4
                error("Not enough data on line $i of particle data!")
            end
            x = parse(Float64, tokens[2])
            y = parse(Float64, tokens[3])
            positions[i] = Vec2(x, y)
            diameters[i] = parse(Float64, tokens[4]) * 2.0
        end
        return positions, Lx_file, Ly_file, diameters
    end
end

#################################
# Utility: Periodic Wrapping    #
#################################
function apply_periodic!(positions::Vector{Vec2}, params::SimulationParams)
    for pos in positions
        pos[1] = mod(pos[1], params.Lx)
        pos[2] = mod(pos[2], params.Ly)
    end
end

############################################################
# Lees–Edwards Minimum Image Convention for Sheared Systems  #
############################################################
function minimum_image(pos_i::Vec2, pos_j::Vec2, gamma::Float64, params::SimulationParams)
    dx = pos_i[1] - pos_j[1]
    dy = pos_i[2] - pos_j[2]
    n_y = round(Int, dy / params.Ly)
    dy -= n_y * params.Ly
    dx -= gamma * params.Lx * n_y
    dx -= params.Lx * round(dx / params.Lx)
    return Vec2(dx, dy)
end

##########################################
# Pairwise Potential and Force Functions #
##########################################
# Computes the potential energy for a pair separated by distance r,
# using diameters sigma_i and sigma_j.
function pair_potential_energy(r::Float64, sigma_i::Float64, sigma_j::Float64, params::SimulationParams)
    σ_eff = 0.5 * (sigma_i + sigma_j)
    σ_eff *= (1.0 - params.non_additivity * abs(sigma_i - sigma_j))
    if r < params.r_cut * σ_eff
        term_1 = (σ_eff / r)^12
        c0 = -28.0 / (params.r_cut^12)
        c2 = 48.0 / (params.r_cut^14)
        c4 = -21.0 / (params.r_cut^16)
        term_2 = c2 * (r / σ_eff)^2
        term_3 = c4 * (r / σ_eff)^4
        return term_1 + c0 + term_2 + term_3
    else
        return 0.0
    end
end

# Computes the pair force vector given displacement r_vec and distance r.
function pair_force(r_vec::Vec2, r::Float64, sigma_i::Float64, sigma_j::Float64, params::SimulationParams)
    σ_eff = 0.5 * (sigma_i + sigma_j)
    σ_eff *= (1.0 - params.non_additivity * abs(sigma_i - sigma_j))
    if r < params.r_cut * σ_eff
        c2 = 48.0 / (params.r_cut^14)
        c4 = -21.0 / (params.r_cut^16)
        # term_1 = (σ_eff / r)^12 (its derivative gives -12 σ_eff^12 / r^13)
        # Derivative of term_2: 2*c2*(r/σ_eff)^2/ r?  Let's compute explicitly:
        # d/dr ( (σ_eff/r)^12 ) = -12*σ_eff^12 / r^13,
        # d/dr (c2*(r/σ_eff)^2) = 2*c2*r/σ_eff^2,
        # d/dr (c4*(r/σ_eff)^4) = 4*c4*r^3/σ_eff^4.
        force_mag = 12.0 * σ_eff^12 / r^13 - 2.0 * c2 * r / (σ_eff^2) - 4.0 * c4 * r^3 / (σ_eff^4)
        return (force_mag * r_vec) / r
    else
        return Vec2(0.0, 0.0)
    end
end

###############################
# Cell List Construction      #
###############################
function build_cell_list(positions::Vector{Vec2}, params::SimulationParams)
    n_cells_x = max(Int(fld(params.Lx, params.r_cut)), 1)
    n_cells_y = max(Int(fld(params.Ly, params.r_cut)), 1)
    cell_size_x = params.Lx / n_cells_x
    cell_size_y = params.Ly / n_cells_y
    cell_list = [Int[] for i in 1:n_cells_x, j in 1:n_cells_y]
    for (i, pos) in enumerate(positions)
        cx = mod(floor(Int, pos[1] / cell_size_x), n_cells_x) + 1
        cy = mod(floor(Int, pos[2] / cell_size_y), n_cells_y) + 1
        push!(cell_list[cx, cy], i)
    end
    return cell_list, n_cells_x, n_cells_y, cell_size_x, cell_size_y
end

##########################################
# Compute Forces and Total Energy        #
##########################################
# Now the diameters vector is passed along.
function compute_forces(positions::Vector{Vec2}, diameters::Vector{Float64},
    gamma::Float64, params::SimulationParams)
    Np = length(positions)
    forces = [Vec2(0.0, 0.0) for _ in 1:Np]
    energy = 0.0
    cell_list, n_cells_x, n_cells_y, _, _ = build_cell_list(positions, params)
    for cx in 1:n_cells_x
        for cy in 1:n_cells_y
            cell_particles = cell_list[cx, cy]
            for i_idx in eachindex(cell_particles)
                i = cell_particles[i_idx]
                for dx in -1:1
                    for dy in -1:1
                        ncx = mod(cx - 1 + dx, n_cells_x) + 1
                        ncy = mod(cy - 1 + dy, n_cells_y) + 1
                        for j in cell_list[ncx, ncy]
                            if (ncx == cx && ncy == cy && j <= i)
                                continue
                            end
                            disp = minimum_image(positions[i], positions[j], gamma, params)
                            r = norm(disp)
                            if r < params.r_cut
                                energy += pair_potential_energy(r, diameters[i], diameters[j], params)
                                fpair = pair_force(disp, r, diameters[i], diameters[j], params)
                                forces[i] += fpair
                                forces[j] -= fpair
                            end
                        end
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
# Virial stress: σ = (1/V) ∑_{i<j} r_ij ⊗ f_ij.
function compute_stress_tensor(positions::Vector{Vec2}, diameters::Vector{Float64},
    gamma::Float64, params::SimulationParams)
    V = params.Lx * params.Ly
    stress = zeros(2, 2)
    cell_list, n_cells_x, n_cells_y, _, _ = build_cell_list(positions, params)
    for cx in 1:n_cells_x
        for cy in 1:n_cells_y
            cell_particles = cell_list[cx, cy]
            for i_idx in eachindex(cell_particles)
                i = cell_particles[i_idx]
                for dx in -1:1
                    for dy in -1:1
                        ncx = mod(cx - 1 + dx, n_cells_x) + 1
                        ncy = mod(cy - 1 + dy, n_cells_y) + 1
                        for j in cell_list[ncx, ncy]
                            if (ncx == cx && ncy == cy && j <= i)
                                continue
                            end
                            disp = minimum_image(positions[i], positions[j], gamma, params)
                            r = norm(disp)
                            if r < params.r_cut
                                fpair = pair_force(disp, r, diameters[i], diameters[j], params)
                                stress .+= disp * transpose(fpair)
                            end
                        end
                    end
                end
            end
        end
    end
    stress ./= V
    return stress
end

##########################################
# Plastic Event Detection                #
##########################################
# Returns true if (ΔE/Δγ) falls below the threshold.
function plastic_event_detected(e_prev::Float64, e_current::Float64,
    dgamma::Float64, threshold::Float64)
    dE_dgamma = (e_current - e_prev) / dgamma
    return dE_dgamma < threshold
end

##########################################################################
# Helper: Evaluate candidate point along the search direction
##########################################################################
function evaluate_candidate(α::Float64, x_old::Vector{Vec2}, d::Vector{Vec2},
    diameters::Vector{Float64}, gamma::Float64,
    params::SimulationParams)
    # Compute candidate positions: x_candidate = x_old + α * d
    candidate = [x_old[i] + α * d[i] for i in 1:length(x_old)]
    apply_periodic!(candidate, params)
    forces, E = compute_forces(candidate, diameters, gamma, params)
    # Gradient is minus the forces.
    candidate_grad = [-f for f in forces]
    return candidate, E, candidate_grad
end

##########################################################################
# Helper: Zoom procedure to find an acceptable α in [α_lo, α_hi]
##########################################################################
function zoom(α_lo::Float64, α_hi::Float64, x_old::Vector{Vec2}, d::Vector{Vec2},
    E_current::Float64, d_dot_g_current::Float64,
    diameters::Vector{Float64}, gamma::Float64, params::SimulationParams,
    c1::Float64, c2::Float64)
    max_zoom_iter = 20
    candidate = Vector{Vec2}()
    E_candidate = Inf
    candidate_grad = Vector{Vec2}()
    α_j = 0.0
    for iter in 1:max_zoom_iter
        α_j = (α_lo + α_hi) / 2.0
        candidate, E_candidate, candidate_grad = evaluate_candidate(α_j, x_old, d, diameters, gamma, params)
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
# Wolfe Conditions Based Line Search
##########################################################################
function line_search_wolfe!(x_old::Vector{Vec2}, diameters::Vector{Float64},
    gamma::Float64, params::SimulationParams,
    d::Vector{Vec2}, E_current::Float64,
    g_current::Vector{Vec2}, c1::Float64, c2::Float64)
    # Compute directional derivative at the starting point:
    d_dot_g_current = sum(dot(d[i], g_current[i]) for i in 1:length(d))
    α_prev = 0.0
    α = 1.0  # initial trial step
    candidate, E_candidate, candidate_grad = evaluate_candidate(α, x_old, d, diameters, gamma, params)
    max_iter = 20
    for iter in 1:max_iter
        if (E_candidate > E_current + c1 * α * d_dot_g_current) ||
           (iter > 1 && E_candidate >= evaluate_candidate(α_prev, x_old, d, diameters, gamma, params)[2])
            # If not, zoom between α_prev and α.
            return zoom(α_prev, α, x_old, d, E_current, d_dot_g_current, diameters, gamma, params, c1, c2)
        end
        d_dot_g_candidate = sum(dot(d[i], candidate_grad[i]) for i in 1:length(d))
        if abs(d_dot_g_candidate) <= -c2 * d_dot_g_current
            return α, candidate, E_candidate, candidate_grad
        end
        if d_dot_g_candidate >= 0
            return zoom(α, α_prev, x_old, d, E_current, d_dot_g_current, diameters, gamma, params, c1, c2)
        end
        α_prev = α
        α *= 2.0  # increase step size
        candidate, E_candidate, candidate_grad = evaluate_candidate(α, x_old, d, diameters, gamma, params)
    end
    return α, candidate, E_candidate, candidate_grad
end

##########################################################################
# Conjugate Gradient Minimization (Polak–Ribiere, Wolfe line search)
##########################################################################
function conjugate_gradient_minimization_wolfe!(positions::Vector{Vec2},
    diameters::Vector{Float64},
    gamma::Float64, params::SimulationParams)
    Np = length(positions)
    # Compute initial forces and energy.
    forces, energy = compute_forces(positions, diameters, gamma, params)
    # Gradient: g = -forces.
    g = [-f for f in forces]
    # Initial search direction: steepest descent.
    d = [-g_i for g_i in g]

    tol = params.fire_tol  # use same tolerance as before
    max_iter = params.fire_max_steps
    iter = 0
    # Wolfe parameters: you can adjust these.
    c1 = 1e-4
    c2 = 0.9

    # Save current positions as x_old.
    x_old = [copy(positions[i]) for i in 1:Np]
    while iter < max_iter
        iter += 1
        # Check convergence: norm of gradient.
        g_norm = sqrt(sum(norm(gi)^2 for gi in g))
        if g_norm < tol
            println("CG converged in $iter iterations; gradient norm = $g_norm")
            return energy
        end

        # Directional derivative.
        d_dot_g = sum(dot(d[i], g[i]) for i in 1:Np)
        if d_dot_g >= 0
            # Not a descent direction; reset.
            d = [-g_i for g_i in g]
            d_dot_g = sum(dot(d[i], g[i]) for i in 1:Np)
        end

        # --- Wolfe line search ---
        α, candidate, E_candidate, candidate_grad = line_search_wolfe!(x_old, diameters, gamma, params, d, energy, g, c1, c2)
        # Update positions to candidate.
        for i in 1:Np
            positions[i] = candidate[i]
        end
        # Update energy.
        energy = E_candidate

        # Compute new gradient.
        g_new = candidate_grad

        # Compute Polak–Ribiere coefficient.
        num = 0.0
        den = 0.0
        for i in 1:Np
            num += dot(g_new[i] - g[i], g_new[i])
            den += dot(g[i], g[i])
        end
        β = max(0.0, num / den)

        # Update search direction.
        for i in 1:Np
            d[i] = -g_new[i] + β * d[i]
        end

        # Update the old positions and gradient.
        x_old = [copy(positions[i]) for i in 1:Np]
        g = g_new

        if iter % 100 == 0
            println("CG iteration $iter: gradient norm = $g_norm, energy = $energy")
        end
    end

    println("CG reached maximum iterations ($max_iter); gradient norm = $(sqrt(sum(norm(gi)^2 for gi in g)))")
    return energy
end


##############################################
# Main Simulation: Athermal Quasistatic Shear #
##############################################
# If a filename is provided, load positions and diameters from file.
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
        # Generate a random configuration and assign all diameters = 1.0.
        params.N = params.N  # use default N from params
        positions = [Vec2(rand() * params.Lx, rand() * params.Ly) for _ in 1:params.N]
        diameters = ones(Float64, params.N)
    end

    # Initial energy minimization (γ = 0).
    gamma = 0.0
    println("Performing initial energy minimization (γ = $gamma)...")
    e_prev = conjugate_gradient_minimization_wolfe!(positions, diameters, gamma, params)
    e_prev /= params.N
    println("γ = $gamma, Energy = $e_prev")
    println("Initial Stress tensor:")
    println(compute_stress_tensor(positions, diameters, gamma, params))

    step = 0
    while true
        step += 1
        # Apply affine shear: x' = x + dγ·y.
        for pos in positions
            pos[1] += params.dgamma * pos[2]
        end
        gamma += params.dgamma
        apply_periodic!(positions, params)
        e_current = conjugate_gradient_minimization_wolfe!(positions, diameters, gamma, params)
        e_current /= params.N
        println("Step $step: γ = $gamma, Energy = $e_current")
        println("Stress tensor:")
        println(compute_stress_tensor(positions, diameters, gamma, params))
        if plastic_event_detected(e_prev, e_current, params.dgamma, params.plastic_threshold)
            println("Plastic event detected at γ = $gamma (step $step)!")
            break
        end
        e_prev = e_current
    end
end

###########################
# Run the Simulation      #
###########################
# To run with a configuration file:
#    run_athermal_quasistatic("configuration_file.txt")
#
# To run with a random configuration:
run_athermal_quasistatic("snapshot_step_500000.xyz")
