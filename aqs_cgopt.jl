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
    Lx::Float64           # Box length in x
    Ly::Float64           # Box length in y
    N::Int                # Number of particles
    r_cut::Float64        # Interaction cutoff (in units of effective diameter)
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
        1.25,      # r_cut
        0.01,      # dt_initial
        0.01,      # dt_max
        1.02,       # f_inc
        0.5,       # f_dec
        0.1,       # alpha0
        1e-5,      # dgamma (strain increment)
        1e-6,      # fire_tol
        100000,    # fire_max_steps
        -1e-6,       # plastic_threshold (plastic event if ΔE/Δγ < threshold)
        0.2,        # non_additivity
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

##########################################
# FIRE Energy Minimization Algorithm     #
##########################################
function fire_minimization!(
    positions::Vector{Vec2},
    diameters::Vector{Float64},
    gamma::Float64,
    params::SimulationParams,
)
    Np = length(positions)
    v = [Vec2(0.0, 0.0) for _ in 1:Np]
    dt = params.dt_initial
    α = params.alpha0

    no_progress_limit = 50
    no_progress_counter = 0
    best_F_norm = Inf
    # Use a variable to check convergence
    convergence = false

    for _ in 1:(params.fire_max_steps)
        forces, energy = compute_forces(positions, diameters, gamma, params)

        F_norm = sqrt(sum(norm(f)^2 for f in forces))

        if F_norm < params.fire_tol
            convergence = true
            return energy, convergence
        end

        if F_norm < best_F_norm * 0.99
            best_F_norm = F_norm
            no_progress_counter = 0
        else
            no_progress_counter += 1
        end

        if no_progress_counter >= no_progress_limit
            for i in 1:Np
                v[i] = Vec2(0.0, 0.0)
            end
            dt = params.dt_initial
            no_progress_counter = 0
        end

        for i in 1:Np
            v[i] += dt * forces[i]
        end

        P = sum(dot(v[i], forces[i]) for i in 1:Np)

        if P > 0
            v_norm = sqrt(sum(norm(v[i])^2 for i in 1:Np))
            f_norm = sqrt(sum(norm(forces[i])^2 for i in 1:Np))
            if v_norm > 0 && f_norm > 0
                for i in 1:Np
                    v[i] = (1 - α) * v[i] + α * (v_norm / f_norm) * forces[i]
                end
            end
            dt = min(dt * params.f_inc, params.dt_max)
            α *= 0.99
        else
            dt *= params.f_dec
            for i in 1:Np
                v[i] = Vec2(0.0, 0.0)
            end
            α = params.alpha0
        end

        for i in 1:Np
            positions[i] += dt * v[i]
        end
        apply_periodic!(positions, gamma, params)
    end

    forces, energy = compute_forces(positions, diameters, gamma, params)
    F_norm = sqrt(sum(norm(f)^2 for f in forces))

    @warn "FIRE did not converge after $(params.fire_max_steps) steps; final F_norm = $(F_norm)"

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
    (e_prev, convergence) = fire_minimization!(positions, diameters, gamma, params)
    # Check if FIRE converged
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
    energy_file = open("energy_aqs.txt", "w")
    stress_file = open("stress_aqs.txt", "w")

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
        (e_current, convergence) = fire_minimization!(positions, diameters, gamma, params)
        # Check if FIRE converged
        if !convergence
            @error "FIRE did not converge at γ = $gamma!"
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
