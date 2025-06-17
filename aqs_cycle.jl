using LinearAlgebra
using Random
using StaticArrays
using DelimitedFiles
using Printf

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
        0.01,      # dt_max
        1.1,       # f_inc
        0.5,       # f_dec
        0.1,       # alpha0
        5,
        1e-4,      # dgamma (strain increment)
        1e-8,      # fire_tol
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

"""
This functions read a configuration file with the following format:

2000
Lattice="44.721359549995796 0 0 44.721359549995796" Properties=species:S:1:pos:R:3:radius:R:1 step=1000000
A 0.7929271081743389 0.5983718258660623 0.0 0.42053595688948275

The first line contains the number of particles.
The second line contains the lattice information and properties.
The subsequent lines contain the particle data, with species, position (x, y, z = 0), and radius.
"""
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
        # Notice that we only have 4 elements in the lattice string for 2D.
        if length(lattice_tokens) < 4
            error("Unexpected lattice format!")
        end
        # For 2D, assume Lx is token 1 and Ly is token 4
        Lx_file = parse(Float64, lattice_tokens[1])
        Ly_file = parse(Float64, lattice_tokens[4])

        # Initialize arrays.
        positions = Vector{Vec2}(undef, N_particles)
        diameters = Vector{Float64}(undef, N_particles)

        # Read particle data.
        for i in 1:N_particles
            line = readline(io)
            tokens = split(strip(line))
            # Here we expect species, radius, x, y and z = 0, so 5 in total
            if length(tokens) < 5
                error("Not enough data on line $i of particle data!")
            end
            # Load the positions
            x = parse(Float64, tokens[2])
            y = parse(Float64, tokens[3])
            positions[i] = Vec2(x, y)
            # The file gives radii; convert to diameter.
            diameters[i] = parse(Float64, tokens[5]) * 2.0
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
    n_cells_y = ceil(Int, p.Ly / max_r_dist)
    n_cells_x = ceil(Int, p.Lx / max_r_dist)
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
                    j <= i && continue
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
                    j <= i && continue
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
"""
    run_cyclic_shear(
      filename::String;
      save_dirname::String="cyclic_results",
      save_config::Bool=true,
      n_cycles::Int=5,
      gamma_max::Float64=0.2,
    )

Perform an AQS cyclic shear protocol for `n_cycles` full cycles, each consisting of four ramps:

  1)  0 → –γ_max  
  2) –γ_max →  0  
  3)  0 → +γ_max  
  4) +γ_max →  0  

Results for each cycle are stored under:
  save_dirname/cycle_001/, cycle_002/, …
Each directory contains a `results.txt` with columns
  cycle  leg  step  γ       σ_xy        E_per_particle
and, if `save_config`, per‐step configuration snapshots.
"""
function run_cyclic_shear(
    filename::String;
    save_dirname::String="cyclic_results",
    save_config::Bool=true,
    n_cycles::Int=5,
    gamma_max::Float64=0.2,
)
    # ——— 1) Read input + initial minimization ———
    params = default_params()
    positions, Lx_file, Ly_file, diameters = read_configuration(filename)
    params.Lx, params.Ly, params.N = Lx_file, Ly_file, length(positions)

    println("Loaded N=$(params.N) from $filename; box=($Lx_file, $Ly_file)")
    gamma = 0.0
    println("Initial minimization at γ=0.0 …")
    e_prev, converged = fire_minimization!(positions, diameters, gamma, params)
    @assert converged "FIRE minimization failed at γ=0!"
    e_prev /= params.N

    # ——— 2) Make base directory ———
    mkpath(save_dirname)

    # ——— 3) Loop over cycles ———
    for cycle in 1:n_cycles
        println("=== Starting cycle $cycle / $n_cycles ===")
        cycle_dir = joinpath(save_dirname, @sprintf("cycle_%03d", cycle))
        mkpath(cycle_dir)

        # Open results file for this cycle
        res_path = joinpath(cycle_dir, "results.txt")
        open(res_path, "w") do io
            # Header
            println(io, "# cycle  leg  step    γ       σ_xy        E_per_particle")

            # Optionally save the γ=0 config at start of each cycle
            if save_config
                start_conf = joinpath(
                    cycle_dir, @sprintf("conf_cycle%03d_γ%.4g.xyz", cycle, gamma)
                )
                save_configuration(start_conf, positions, diameters, params)
            end

            # Define the four targets for each cycle
            targets = (-gamma_max, 0.0, +gamma_max, 0.0)

            for (leg, target) in enumerate(targets)
                direction = sign(target - gamma)  # ±1 or 0
                step = 0

                # Ramp from current γ toward `target`
                while abs(target - gamma) > 1e-12
                    step += 1

                    # Affine shear: x_i += direction * dγ * y_i
                    for pos in positions
                        pos[1] += direction * params.dgamma * pos[2]
                    end
                    gamma += direction * params.dgamma
                    apply_periodic!(positions, gamma, params)

                    # Energy minimization + stress
                    e_cur, converged = fire_minimization!(
                        positions, diameters, gamma, params
                    )
                    @assert converged "FIRE failed at γ=$gamma (cycle $cycle, leg $leg)"
                    e_cur /= params.N

                    σ = compute_stress_tensor(positions, diameters, gamma, params)[1, 2]

                    # Record to file
                    @printf(
                        io,
                        "%5d   %1d   %5d   %8.5f   %10.5e   %10.5e\n",
                        cycle,
                        leg,
                        step,
                        gamma,
                        σ,
                        e_cur
                    )

                    # Save snapshot if desired
                    if save_config
                        fname = @sprintf(
                            "conf_cycle%03d_leg%d_step%04d_γ%.4g.xyz",
                            cycle,
                            leg,
                            step,
                            gamma
                        )
                        save_configuration(
                            joinpath(cycle_dir, fname), positions, diameters, params
                        )
                    end

                    e_prev = e_cur
                end
            end
        end

        println("=== Finished cycle $cycle; final γ=$(round(gamma, sigdigits=6)) ===")
    end

    return nothing
end

# Example entry point
function main()
    input_xyz = "init.xyz"
    return run_cyclic_shear(
        input_xyz; save_dirname="cyclic_aqs", save_config=true, n_cycles=10, gamma_max=0.2
    )
end

main()
