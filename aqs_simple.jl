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
    Nmin::Float64
    dgamma::Float64       # Strain increment per shear step
    fire_tol::Float64     # FIRE convergence tolerance
    fire_max_steps::Int   # Maximum FIRE iterations per shear step
    plastic_threshold::Float64  # Threshold for (ΔE/Δγ) to detect a plastic event
    non_additivity::Float64     # Non-additivity parameter for the effective diameter
    c0::Float64
    c2::Float64
    c4::Float64
end

# Default parameters.
function default_params()
    default_r_cut = 1.25
    c0 = -28.0 / default_r_cut^12
    c2 = 48.0 / default_r_cut^14
    c4 = -21.0 / default_r_cut^16
    return SimulationParams(
        10.0,      # Lx (will be overwritten if configuration file is used)
        10.0,      # Ly
        100,       # N (will be overwritten if configuration file is used)
        default_r_cut,      # r_cut
        0.005,      # dt_initial
        0.02,      # dt_max
        1.1,       # f_inc
        0.5,       # f_dec
        0.1,       # alpha0
        5,
        1e-4,      # dgamma (strain increment)
        1e-6,      # fire_tol
        100000,    # fire_max_steps
        -1e-6,       # plastic_threshold (plastic event if ΔE/Δγ < threshold)
        0.2,        # non_additivity
        c0,
        c2,
        c4,
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

function read_configuration_alt(filename::String)
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
            if length(tokens) < 4
                error("Not enough data on line $i of particle data!")
            end
            # The file gives radii; convert to diameter.
            diameters[i] = parse(Float64, tokens[4]) * 2.0
            # Load the positions
            x = parse(Float64, tokens[2])
            y = parse(Float64, tokens[3])
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

##########################################
# Compute Forces and Total Energy        #
##########################################
function compute_forces(
    positions::Vector{Vec2}, diameters::Vector{Float64}, γ::Float64, p::SimulationParams
)
    Np = length(positions)
    forces = [Vec2(0.0, 0.0) for _ in 1:Np]
    energy = 0.0

    @inbounds for i in 1:(Np - 1)
        for j in (i + 1):Np
            # minimum-image displacement
            disp = minimum_image(positions[i], positions[j], γ, p)
            r = norm(disp)
            σ_i, σ_j = diameters[i], diameters[j]
            σ_eff = 0.5 * (σ_i + σ_j) * (1.0 - p.non_additivity * abs(σ_i - σ_j))
            if r < p.r_cut * σ_eff
                energy += pair_potential_energy(r, σ_eff, p)
                fpair = pair_force(disp, r, σ_eff, p)
                forces[i] += fpair
                forces[j] -= fpair
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
    Np = length(positions)

    @inbounds for i in 1:(Np - 1)
        for j in (i + 1):Np
            disp = minimum_image(positions[i], positions[j], γ, p)
            r = norm(disp)
            σ_i, σ_j = diameters[i], diameters[j]
            σ_eff = 0.5 * (σ_i + σ_j) * (1.0 - p.non_additivity * abs(σ_i - σ_j))
            if r < p.r_cut * σ_eff
                fpair = pair_force(disp, r, σ_eff, p)
                # stress contribution: – disp ⊗ f
                stress .-= disp * transpose(fpair)
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

##########################################
# FIRE Energy Minimization Algorithm     #
##########################################
function fire_minimization!(
    positions::Vector{Vec2},
    diameters::Vector{Float64},
    gamma::Float64,
    params::SimulationParams,
)
    v = [Vec2(0.0, 0.0) for _ in eachindex(positions)]
    dt = params.dt_initial
    α = params.alpha0
    steps_since_neg = 0
    ndof = 2.0 * (params.N - 1.0)  # Number of degrees of freedom (2D system)

    # Use a variable to check convergence
    convergence = false

    for step in 1:(params.fire_max_steps)
        forces, energy = compute_forces(positions, diameters, gamma, params)

        F_norm = sqrt(sum(norm(f)^2 for f in forces))

        if mod(step, 100) == 0
            @info "FIRE step $step: F_norm = $(F_norm / sqrt(ndof)), dt = $dt"
        end

        if F_norm / sqrt(ndof) < params.fire_tol
            convergence = true
            return energy, convergence
        end

        for i in eachindex(v)
            v[i] += dt * forces[i]
        end

        P = sum(dot(v[i], forces[i]) for i in eachindex(v))
        v_norm = sqrt(sum(norm(vi)^2 for vi in v))
        f_norm = sqrt(sum(norm(f)^2 for f in forces))
        if v_norm > 0 && f_norm > 0
            scale = α * (v_norm / f_norm)
            for i in eachindex(v)
                v[i] = (1 - α) * v[i] + scale * forces[i]
            end
        end

        if P > 0
            steps_since_neg += 1
            if steps_since_neg > params.Nmin
                dt = min(dt * params.f_inc, params.dt_max)
                α *= 0.99
            end
        else
            dt = max(dt * params.f_dec, params.dt_initial)
            fill!(v, Vec2(0.0, 0.0))
            α = params.alpha0
            steps_since_neg = 0
        end

        for i in eachindex(positions)
            positions[i] += dt * v[i]
        end
        apply_periodic!(positions, gamma, params)
    end

    forces, energy = compute_forces(positions, diameters, gamma, params)
    F_norm = sqrt(sum(norm(f)^2 for f in forces))
    F_norm /= sqrt(ndof)

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
        for i in eachindex(positions)
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
    gamma_max = 0.2
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

    # Create a directory to save everything
    save_dir = mkpath("aqs_results_simple")

    # Save the initial configuration.
    save_configuration("initial_configuration.xyz", positions, diameters, params)

    # Let's open a file to save the energy information at every step
    energy_file = open(joinpath(save_dir, "energy_aqs.txt"), "w")
    stress_file = open(joinpath(save_dir, "stress_aqs.txt"), "w")

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
            @error "FIRE did not converge at γ = $gamma"
            @info "Stopping the simulation."
            break
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

        # save_filename = joinpath(save_dir, @sprintf("conf_%.4g.xyz", gamma))
        # save_configuration(save_filename, positions, diameters, params)
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
