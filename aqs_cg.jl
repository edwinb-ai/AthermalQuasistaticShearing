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
        1e-4,      # dgamma (strain increment)
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
# Wolfe Conditions Based Line Search
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
    forces, energy = compute_forces(positions, diameters, gamma, params)
    # Gradient: g = -forces.
    g = [-f for f in forces]
    # Initial search direction: steepest descent.
    d = [-g_i for g_i in g]

    no_progress_limit = 50
    no_progress_counter = 0
    best_gradient_norm = Inf
    # Use a variable to check convergence
    convergence = false

    # Wolfe parameters
    c1 = 1e-4
    c2 = 0.9

    # Save current positions as x_old.
    x_old = [copy(positions[i]) for i in 1:Np]

    for iter in 1:(params.cg_max_steps)
        # Check convergence: norm of gradient (force).
        gradient_norm = sqrt(sum(norm(gi)^2 for gi in g))

        if gradient_norm < params.cg_tol
            convergence = true
            return energy, gradient_norm, convergence
        end

        if gradient_norm < best_gradient_norm * 0.99
            best_gradient_norm = gradient_norm
            no_progress_counter = 0
        else
            no_progress_counter += 1
        end

        if no_progress_counter >= no_progress_limit
            # Reset to steepest descent
            d = [-g_i for g_i in g]
            no_progress_counter = 0
        end

        # Check if search direction is a descent direction
        d_dot_g = sum(dot(d[i], g[i]) for i in 1:Np)
        if d_dot_g >= 0
            # Not a descent direction; reset to steepest descent.
            d = [-g_i for g_i in g]
            d_dot_g = sum(dot(d[i], g[i]) for i in 1:Np)
        end

        # --- Wolfe line search ---
        α, candidate, E_candidate, candidate_grad = line_search_wolfe!(
            x_old, diameters, gamma, params, d, energy, g, c1, c2
        )

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
    end

    forces, energy = compute_forces(positions, diameters, gamma, params)
    gradient_norm = sqrt(sum(norm(f)^2 for f in forces))

    @warn "Conjugate Gradient did not converge after $(params.cg_max_steps) steps; final gradient norm = $(gradient_norm)"

    return energy, gradient_norm, convergence
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
    (e_prev, grad_norm, convergence) = conjugate_gradient_minimization!(
        positions, diameters, gamma, params
    )
    # Check if CG converged
    if !convergence
        @error "Initial energy minimization did not converge!"
        return nothing
    end
    # Normalize the energy per particle.
    e_prev /= params.N
    println("γ = $gamma, Energy per particle = $e_prev, Gradient norm = $grad_norm")
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
        (e_current, grad_norm, convergence) = conjugate_gradient_minimization!(
            positions, diameters, gamma, params
        )
        # Check if CG converged
        if !convergence
            @error "Conjugate Gradient did not converge at γ = $gamma"
            # @info "Halving the strain increment and retrying..."
            # if params.dgamma < gamma_min
            #     @error "Strain increment too small; stopping simulation."
            #     exit(1)
            # end
            # params.dgamma /= 2.0
            # gamma -= params.dgamma  # Roll back the gamma increment
            # step -= 1  # Roll back the step count
            break
        end
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
run_athermal_quasistatic("init.xyz")