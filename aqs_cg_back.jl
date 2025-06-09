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
    cg_tol::Float64                # CG convergence tolerance
    cg_max_steps::Int              # Maximum CG iterations per shear step
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
        1e-5,      # dgamma (strain increment)
        1e-6,      # cg_tol (CG convergence tolerance)
        100000,    # cg_max_steps (CG max iterations)
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
@inline function apply_periodic!(
    positions::Vector{Vec2}, gamma::Float64, params::SimulationParams
)
    @inbounds for pos in positions
        # 1) wrap in y, record how many boxes we moved
        n_y = floor(Int, pos[2] / params.Ly)
        pos[2] -= n_y * params.Ly

        # 2) apply the shear‐offset for that crossing
        pos[1] -= n_y * gamma * params.Lx

        # 3) now wrap x normally
        n_x = floor(Int, pos[1] / params.Lx)
        pos[1] -= n_x * params.Lx
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
function pair_potential_energy(
    r::Float64, sigma_i::Float64, sigma_j::Float64, params::SimulationParams
)
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

function pair_force(
    r_vec::Vec2, r::Float64, sigma_i::Float64, sigma_j::Float64, params::SimulationParams
)
    σ_eff = 0.5 * (sigma_i + sigma_j)
    σ_eff *= (1.0 - params.non_additivity * abs(sigma_i - sigma_j))
    if r < params.r_cut * σ_eff
        c2 = 48.0 / (params.r_cut^14)
        c4 = -21.0 / (params.r_cut^16)
        force_mag =
            12.0 * σ_eff^12 / r^13 - 2.0 * c2 * r / (σ_eff^2) - 4.0 * c4 * r^3 / (σ_eff^4)
        return (force_mag * r_vec) / r
    else
        return Vec2(0.0, 0.0)
    end
end

###############################
# Cell List Construction      #
###############################
function build_cell_list(
    positions::Vector{Vec2}, params::SimulationParams, max_diameter::Float64
)
    # A safe upper bound on σ_eff is (max_diameter) * (1 - δ |diff|),
    # but if you want to be conservative, just use max_diameter.
    max_r_cut_dist = params.r_cut * max_diameter
    n_cells_x = max(Int(floor(params.Lx / max_r_cut_dist)), 1)
    n_cells_y = max(Int(floor(params.Ly / max_r_cut_dist)), 1)
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
function compute_forces(
    positions::Vector{Vec2},
    diameters::Vector{Float64},
    gamma::Float64,
    params::SimulationParams,
)
    Np = length(positions)
    forces = [Vec2(0.0, 0.0) for _ in 1:Np]
    energy = 0.0
    cell_list, n_cells_x, n_cells_y, _, _ = build_cell_list(
        positions, params, maximum(diameters)
    )
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
                            sigma_i = diameters[i]
                            sigma_j = diameters[j]
                            σ_eff = 0.5 * (sigma_i + sigma_j)
                            σ_eff *= (1.0 - params.non_additivity * abs(sigma_i - sigma_j))
                            if r < params.r_cut * σ_eff
                                energy += pair_potential_energy(r, sigma_i, sigma_j, params)
                                fpair = pair_force(disp, r, sigma_i, sigma_j, params)
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
function compute_stress_tensor(
    positions::Vector{Vec2},
    diameters::Vector{Float64},
    gamma::Float64,
    params::SimulationParams,
)
    V = params.Lx * params.Ly
    stress = zeros(2, 2)
    cell_list, n_cells_x, n_cells_y, _, _ = build_cell_list(
        positions, params, maximum(diameters)
    )
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
                            sigma_i = diameters[i]
                            sigma_j = diameters[j]
                            σ_eff = 0.5 * (sigma_i + sigma_j)
                            σ_eff *= (1.0 - params.non_additivity * abs(sigma_i - sigma_j))
                            if r < params.r_cut * σ_eff
                                fpair = pair_force(disp, r, sigma_i, sigma_j, params)
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
function plastic_event_detected(
    e_prev::Float64, e_current::Float64, dgamma::Float64, threshold::Float64
)
    dE_dgamma = (e_current - e_prev) / dgamma
    return dE_dgamma < threshold
end

##########################################################################
# Helper: Evaluate energy and gradient at a point
##########################################################################
function evaluate_point(
    positions::Vector{Vec2},
    diameters::Vector{Float64},
    gamma::Float64,
    params::SimulationParams,
)
    forces, energy = compute_forces(positions, diameters, gamma, params)
    # Gradient is minus the forces.
    gradient = [-f for f in forces]
    return energy, gradient
end

##########################################################################
# Backtracking Line Search (Armijo condition)
##########################################################################
function backtracking_line_search!(
    positions::Vector{Vec2},
    diameters::Vector{Float64},
    gamma::Float64,
    params::SimulationParams,
    search_direction::Vector{Vec2},
    current_energy::Float64,
    current_gradient::Vector{Vec2},
)
    # Backtracking parameters
    c1 = 1e-4          # Armijo parameter (sufficient decrease)
    ρ = 0.5            # Step reduction factor
    α_max = 1.0        # Initial step size
    α_min = 1e-12      # Minimum step size

    # Compute directional derivative
    directional_derivative = sum(
        dot(search_direction[i], current_gradient[i]) for i in 1:length(search_direction)
    )

    # If not a descent direction, return minimal step
    if directional_derivative >= 0
        return α_min, positions, current_energy, current_gradient
    end

    α = α_max
    candidate_positions = copy(positions)

    # Backtracking loop
    while α >= α_min
        # Compute candidate position: x_new = x_old + α * search_direction
        for i in 1:length(positions)
            candidate_positions[i] = positions[i] + α * search_direction[i]
        end
        apply_periodic!(candidate_positions, gamma, params)

        # Evaluate energy at candidate point
        candidate_energy, candidate_gradient = evaluate_point(
            candidate_positions, diameters, gamma, params
        )

        # Check Armijo condition: f(x + α*d) ≤ f(x) + c1*α*∇f(x)ᵀd
        armijo_threshold = current_energy + c1 * α * directional_derivative

        if candidate_energy <= armijo_threshold
            # Accept this step size
            return α, candidate_positions, candidate_energy, candidate_gradient
        end

        # Reduce step size and try again
        α *= ρ
    end

    # If we reach here, even the minimal step failed
    # Return minimal step anyway (better than not moving)
    for i in 1:length(positions)
        candidate_positions[i] = positions[i] + α_min * search_direction[i]
    end
    apply_periodic!(candidate_positions, gamma, params)
    candidate_energy, candidate_gradient = evaluate_point(
        candidate_positions, diameters, gamma, params
    )

    return α_min, candidate_positions, candidate_energy, candidate_gradient
end

##########################################
# Conjugate Gradient Energy Minimization #
##########################################
function conjugate_gradient_minimization!(
    positions::Vector{Vec2},
    diameters::Vector{Float64},
    gamma::Float64,
    params::SimulationParams,
)
    Np = length(positions)

    # Compute initial forces and energy.
    energy, gradient = evaluate_point(positions, diameters, gamma, params)

    # Initial search direction: steepest descent.
    search_direction = [-g for g in gradient]

    no_progress_limit = 50
    no_progress_counter = 0
    best_gradient_norm = Inf
    convergence = false

    for iter in 1:(params.cg_max_steps)
        # Check convergence: norm of gradient.
        gradient_norm = sqrt(sum(norm(g)^2 for g in gradient))

        if gradient_norm < params.cg_tol
            convergence = true
            return energy, convergence
        end

        # Track progress
        if gradient_norm < best_gradient_norm * 0.99
            best_gradient_norm = gradient_norm
            no_progress_counter = 0
        else
            no_progress_counter += 1
        end

        # Reset to steepest descent if no progress
        if no_progress_counter >= no_progress_limit
            search_direction = [-g for g in gradient]
            no_progress_counter = 0
        end

        # Check if search direction is a descent direction
        directional_derivative = sum(dot(search_direction[i], gradient[i]) for i in 1:Np)
        if directional_derivative >= 0
            # Reset to steepest descent
            search_direction = [-g for g in gradient]
        end

        # Perform backtracking line search
        α, new_positions, new_energy, new_gradient = backtracking_line_search!(
            positions, diameters, gamma, params, search_direction, energy, gradient
        )

        # Update positions and energy
        for i in 1:Np
            positions[i] = new_positions[i]
        end
        energy = new_energy

        # Compute Polak–Ribiere coefficient for next iteration
        numerator = 0.0
        denominator = 0.0
        for i in 1:Np
            numerator += dot(new_gradient[i] - gradient[i], new_gradient[i])
            denominator += dot(gradient[i], gradient[i])
        end
        β = max(0.0, numerator / denominator)

        # Update search direction using Polak-Ribiere formula
        for i in 1:Np
            search_direction[i] = -new_gradient[i] + β * search_direction[i]
        end

        # Update gradient for next iteration
        gradient = new_gradient
    end

    # Final gradient norm check
    final_gradient_norm = sqrt(sum(norm(g)^2 for g in gradient))
    @warn "Conjugate Gradient did not converge after $(params.cg_max_steps) steps; final gradient norm = $(final_gradient_norm)"

    return energy, convergence
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
    gamma_min = 1e-8
    gamma = 0.0
    # Initial energy minimization.
    println("Performing initial energy minimization (γ = $gamma)...")
    (e_prev, convergence) = conjugate_gradient_minimization!(
        positions, diameters, gamma, params
    )
    # Check if CG converged
    if !convergence
        @error "Initial energy minimization did not converge!"
        return nothing
    end
    # Normalize the energy per particle.
    e_prev /= params.N
    println("γ = $gamma, Energy per particle = $e_prev")
    println("Initial Stress tensor:")
    println(compute_stress_tensor(positions, diameters, gamma, params))

    # Save the initial configuration.
    save_configuration("initial_configuration.xyz", positions, diameters, params)

    # Let's open a file to save the energy information at every step
    energy_file = open("energy_aqs_cg.txt", "w")
    stress_file = open("stress_aqs_cg.txt", "w")

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
        (e_current, convergence) = conjugate_gradient_minimization!(
            positions, diameters, gamma, params
        )
        # Check if CG converged
        if !convergence
            @error "Conjugate Gradient did not converge at γ = $gamma!"
            @info "Halving the strain increment and retrying..."
            if params.dgamma < gamma_min
                @error "Strain increment too small; stopping simulation."
                exit(1)
            end
            params.dgamma /= 2.0
            gamma -= params.dgamma  # Roll back the gamma increment
            step -= 1  # Roll back the step count
            continue
        end
        # Normalize the energy per particle.
        e_current /= params.N

        # Write and flush the file
        println(energy_file, e_current)
        flush(energy_file)

        println("Step $step: γ = $gamma, Energy per particle = $e_current")
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

        save_configuration(@sprintf("conf_%.4g.xyz", gamma), positions, diameters, params)
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
run_athermal_quasistatic("initial_poly.xyz")