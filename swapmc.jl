using StaticArrays
using Random
using LinearAlgebra: norm
using Statistics: mean, std

# ============================
# Particle Definition & Utilities
# ============================
struct Particle
    pos::SVector{2,Float64}
    sigma::Float64
end

const non_additivity = 0.2

# Wrap a position into the simulation box.
function apply_pbc(pos::SVector{2,Float64}, L::Float64)
    return SVector(mod(pos[1], L), mod(pos[2], L))
end

# Compute the minimum-image displacement.
function minimum_image(r::SVector{2,Float64}, L::Float64)
    return SVector(r[1] - L * round(r[1] / L), r[2] - L * round(r[2] / L))
end

# -----------------------------
# Effective (Mixed) Diameter Function
# -----------------------------
"""
    effective_sigma(sigma1, sigma2)

Computes the effective (mixed) diameter for two particles with diameters `sigma1` and `sigma2`
including a non-additivity correction.
"""
function effective_sigma(sigma1::Float64, sigma2::Float64)
    σ_eff = 0.5 * (sigma1 + sigma2)
    σ_eff *= (1.0 - non_additivity * abs(sigma1 - sigma2))
    return σ_eff
end

# ============================
# Cell List Data Structure & Functions
# ============================
struct CellList
    cells::Matrix{Vector{Int}}  # cells[i,j] holds a vector of particle indices
    n_cells::Int                # number of cells along each dimension
    cell_size::Float64          # side length of each cell (≈ L / n_cells)
    L::Float64                  # simulation box side length
end

# Determine the cell coordinates (i,j) for a given position.
function cell_coords(pos::SVector{2,Float64}, cell_size::Float64, n_cells::Int)
    ci = floor(Int, pos[1] / cell_size) + 1
    cj = floor(Int, pos[2] / cell_size) + 1
    return (ci, cj)
end

# Build the cell list from all particles.
# The neighbor search cutoff used here is the maximum effective cutoff.
function build_cell_list(particles::Vector{Particle}, L::Float64, cell_cutoff::Float64)
    n_cells = max(1, floor(Int, L / cell_cutoff))
    cell_size = L / n_cells
    cells = [Vector{Int}() for i in 1:n_cells, j in 1:n_cells]
    particle_cells = Vector{Tuple{Int,Int}}(undef, length(particles))
    for i in 1:length(particles)
        pos = particles[i].pos
        (ci, cj) = cell_coords(pos, cell_size, n_cells)
        push!(cells[ci, cj], i)
        particle_cells[i] = (ci, cj)
    end
    return CellList(cells, n_cells, cell_size, L), particle_cells
end

# Get all particle indices in the 3×3 block of cells surrounding (ci, cj) (including itself).
function get_neighbor_indices(cell_list::CellList, ci::Int, cj::Int)
    neighbors = Int[]
    for di in -1:1
        for dj in -1:1
            ni = mod(ci - 1 + di, cell_list.n_cells) + 1
            nj = mod(cj - 1 + dj, cell_list.n_cells) + 1
            append!(neighbors, cell_list.cells[ni, nj])
        end
    end
    return neighbors
end

# Update a single particle's cell membership if it has moved.
function update_particle_cell!(
    cell_list::CellList,
    particle_cells::Vector{Tuple{Int,Int}},
    particles::Vector{Particle},
    i::Int,
)
    new_cell = cell_coords(particles[i].pos, cell_list.cell_size, cell_list.n_cells)
    old_cell = particle_cells[i]
    if new_cell != old_cell
        cell_old = cell_list.cells[old_cell[1], old_cell[2]]
        idx = findfirst(==(i), cell_old)
        if idx !== nothing
            deleteat!(cell_old, idx)
        end
        push!(cell_list.cells[new_cell[1], new_cell[2]], i)
        particle_cells[i] = new_cell
    end

    return nothing
end

# ============================
# Potential & Local Energy Functions
# ============================
"""
    pair_energy(p1, p2, L, r_cut)

Computes the interaction energy between particles `p1` and `p2` using an inverse–power law
potential with a pair–dependent cutoff. The effective cutoff for the pair is
r_cut_eff = r_cut * effective_sigma(p1, p2).
If the separation r exceeds r_cut_eff, the interaction is zero.
"""
function pair_energy(p1::Particle, p2::Particle, L::Float64, r_cut::Float64)
    d = minimum_image(p1.pos - p2.pos, L)
    r = norm(d)
    σ_eff = effective_sigma(p1.sigma, p2.sigma)
    if r >= r_cut * σ_eff
        return 0.0
    else
        # Build the potential term by term.
        term_1 = (σ_eff / r)^12
        c0 = -28.0 / (r_cut^12)
        c2 = 48.0 / (r_cut^14)
        c4 = -21.0 / (r_cut^16)
        term_2 = c2 * (r / σ_eff)^2
        term_3 = c4 * (r / σ_eff)^4
        potential = term_1 + c0 + term_2 + term_3

        return potential
    end
end

# Compute the local energy of particle i (summing interactions with nearby particles).
function local_energy(
    i::Int, particles::Vector{Particle}, cell_list::CellList, r_cut::Float64
)
    pos = particles[i].pos
    (ci, cj) = cell_coords(pos, cell_list.cell_size, cell_list.n_cells)
    neighbors = get_neighbor_indices(cell_list, ci, cj)
    E = 0.0
    for j in neighbors
        if j != i
            E += pair_energy(particles[i], particles[j], cell_list.L, r_cut)
        end
    end
    return E
end

# Compute the local energy for a proposed move of particle i (with a new position).
function local_energy_proposed(
    new_pos::SVector{2,Float64},
    sigma::Float64,
    i::Int,
    particles::Vector{Particle},
    cell_list::CellList,
    r_cut::Float64,
)
    (ci, cj) = cell_coords(new_pos, cell_list.cell_size, cell_list.n_cells)
    neighbors = get_neighbor_indices(cell_list, ci, cj)
    E = 0.0
    for j in neighbors
        if j != i
            d = minimum_image(new_pos - particles[j].pos, cell_list.L)
            r = norm(d)
            σ_eff = effective_sigma(sigma, particles[j].sigma)
            if r < r_cut * σ_eff
                term_1 = (σ_eff / r)^12
                c0 = -28.0 / (r_cut^12)
                c2 = 48.0 / (r_cut^14)
                c4 = -21.0 / (r_cut^16)
                term_2 = c2 * (r / σ_eff)^2
                term_3 = c4 * (r / σ_eff)^4
                potential = term_1 + c0 + term_2 + term_3
                E += potential
            end
        end
    end
    return E
end

# For swap moves: compute the local energy of particle i excluding its interaction with 'exclude'.
function local_energy_excluding(
    i::Int, particles::Vector{Particle}, cell_list::CellList, r_cut::Float64, exclude::Int
)
    (ci, cj) = cell_coords(particles[i].pos, cell_list.cell_size, cell_list.n_cells)
    neighbors = get_neighbor_indices(cell_list, ci, cj)
    E = 0.0
    for j in neighbors
        if j != i && j != exclude
            E += pair_energy(particles[i], particles[j], cell_list.L, r_cut)
        end
    end
    return E
end

# For swap moves: compute the local energy for a temporary state.
function local_energy_custom(
    temp_particle::Particle,
    particles::Vector{Particle},
    cell_list::CellList,
    r_cut::Float64,
    i::Int,
    exclude::Int,
)
    (ci, cj) = cell_coords(temp_particle.pos, cell_list.cell_size, cell_list.n_cells)
    neighbors = get_neighbor_indices(cell_list, ci, cj)
    E = 0.0
    for k in neighbors
        if k != i && k != exclude
            E += pair_energy(temp_particle, particles[k], cell_list.L, r_cut)
        end
    end
    return E
end

# ============================
# Monte Carlo Moves
# ============================
"""
    mc_step!(particles, cell_list, particle_cells, beta, delta, p_swap, r_cut)

Performs one Monte Carlo move. With probability `p_swap` a swap move (exchanging σ values)
is attempted; otherwise a displacement move is attempted.
Returns a tuple `(move_type, accepted)` where `move_type` is either `:swap` or `:disp`.
"""
function mc_step!(
    particles::Vector{Particle},
    cell_list::CellList,
    particle_cells::Vector{Tuple{Int,Int}},
    beta::Float64,
    delta::Float64,
    p_swap::Float64,
    r_cut::Float64,
)
    N = length(particles)
    if rand() < p_swap
        # ----- SWAP MOVE -----
        i = rand(1:N)
        j = rand(1:N)
        while i == j
            j = rand(1:N)
        end

        E_old_i = local_energy_excluding(i, particles, cell_list, r_cut, j)
        E_old_j = local_energy_excluding(j, particles, cell_list, r_cut, i)

        new_sigma_i = particles[j].sigma
        new_sigma_j = particles[i].sigma

        new_particle_i = Particle(particles[i].pos, new_sigma_i)
        new_particle_j = Particle(particles[j].pos, new_sigma_j)

        E_new_i = local_energy_custom(new_particle_i, particles, cell_list, r_cut, i, j)
        E_new_j = local_energy_custom(new_particle_j, particles, cell_list, r_cut, j, i)

        ΔE = (E_new_i + E_new_j) - (E_old_i + E_old_j)
        if rand() < exp(-beta * ΔE)
            particles[i] = Particle(particles[i].pos, new_sigma_i)
            particles[j] = Particle(particles[j].pos, new_sigma_j)
            return (:swap, true)
        else
            return (:swap, false)
        end
    else
        # ----- DISPLACEMENT MOVE -----
        i = rand(1:N)
        old_particle = particles[i]
        E_old = local_energy(i, particles, cell_list, r_cut)
        dx = (rand() - 0.5) * 2 * delta
        dy = (rand() - 0.5) * 2 * delta
        new_pos = old_particle.pos + SVector(dx, dy)
        new_pos = apply_pbc(new_pos, cell_list.L)
        E_new = local_energy_proposed(
            new_pos, old_particle.sigma, i, particles, cell_list, r_cut
        )
        ΔE = E_new - E_old
        if rand() < exp(-beta * ΔE)
            particles[i] = Particle(new_pos, old_particle.sigma)
            update_particle_cell!(cell_list, particle_cells, particles, i)
            return (:disp, true)
        else
            return (:disp, false)
        end
    end
end

# ============================
# Total Energy (Using Cell Lists)
# ============================
function total_energy(particles::Vector{Particle}, cell_list::CellList, r_cut::Float64)
    E = 0.0
    N = length(particles)
    for i in 1:N
        (ci, cj) = cell_coords(particles[i].pos, cell_list.cell_size, cell_list.n_cells)
        neighbors = get_neighbor_indices(cell_list, ci, cj)
        for j in neighbors
            if j > i
                E += pair_energy(particles[i], particles[j], cell_list.L, r_cut)
            end
        end
    end
    return E
end

# ============================
# Function to Adjust Δ (Maximum Displacement)
# ============================
"""
    adjust_delta(delta, disp_accept, disp_attempt; target=0.4)

Adjusts the maximum displacement Δ so that the acceptance ratio for displacement moves
approaches the target (default 40%). If no moves were accepted in the interval, Δ is halved.
"""
function adjust_delta(
    delta::Float64, disp_accept::Int, disp_attempt::Int; target::Float64=0.4
)
    if disp_attempt == 0
        return delta
    end
    ratio = disp_accept / disp_attempt
    if ratio == 0.0
        return delta * 0.5
    else
        return delta * (ratio / target)
    end
end

# ============================
# Snapshot Writer (Extended XYZ Format)
# ============================
"""
    write_snapshot_xyz(filename, particles, step, boxl)

Writes a snapshot of the current system in extended XYZ format. The file will have:
 - The number of particles on the first line.
 - A header (comment) line that includes properties and the current simulation step.
 - One line per particle containing: species, x, y, z (here z is 0.0), and the particle’s radius.
This format can be loaded into OVITO.
"""
function write_snapshot_xyz(
    filename::String, particles::Vector{Particle}, step::Int, boxl::Float64
)
    open(filename, "w") do io
        println(io, length(particles))
        # The header includes a lattice definition (for 2D, z is ignored) and the step number.
        println(
            io,
            "Lattice=\"$(boxl) 0 0 $(boxl)\" Properties=species:S:1:pos:R:3:radius:R:1 step=$step",
        )
        for p in particles
            # Here we use "A" as species and report the radius as sigma/2.
            println(io, "A $(p.pos[1]) $(p.pos[2]) 0.0 $(p.sigma/2.0)")
        end
    end

    return nothing
end

# ============================
# Additional Functions
# ============================
"""
    inverse_diameters(; sigma_min, sigma_max)

Samples a diameter from an inverse power–law probability distribution,
P(σ) = A/σ³, using inverse transform sampling.
"""
function inverse_diameters(; sigma_min=0.73, sigma_max=1.62)
    u = rand()
    term_1 = (1.0 - u) / sigma_min^2
    term_2 = u / sigma_max^2
    return 1.0 / sqrt(term_1 + term_2)
end

function check_polidispersity(particles)
    all_sigmas = [p.sigma for p in particles]
    mean_diameter = mean(all_sigmas)
    stdev_diameter = std(all_sigmas)
    polydispersity = stdev_diameter / mean_diameter
    println("Polydispersity in the system: $(polydispersity)")

    return nothing
end

"""
    initialize_lattice(n_particles, boxl)

Returns two arrays with the x and y positions of particles arranged on a square lattice.
This is used to initialize the system in an overlap–free configuration.
"""
function initialize_lattice(n_particles::Int, boxl::Float64)
    side_length = ceil(Int, sqrt(n_particles))
    spacing = boxl / side_length
    x_positions = Float64[]
    y_positions = Float64[]
    for i in 1:side_length
        for j in 1:side_length
            if length(x_positions) < n_particles
                push!(x_positions, (i - 0.5) * spacing)
                push!(y_positions, (j - 0.5) * spacing)
            else
                break
            end
        end
    end
    return x_positions, y_positions
end

# ============================
# Simulation Runner
# ============================
"""
    run_simulation(N, L, beta, delta, p_swap, nsteps, r_cut; print_interval=1000)

Runs the Monte Carlo simulation.
- N: Number of particles.
- L: Box side length.
- beta: Inverse temperature.
- delta: Initial maximum displacement.
- p_swap: Probability for a swap move.
- nsteps: Total number of Monte Carlo steps.
- r_cut: Base cutoff distance used in the potential.
- print_interval (keyword): Frequency (in steps) to print diagnostics, log energy, and write a snapshot.
Returns the final particle configuration and an array of per–particle energies.
"""
function run_simulation(
    N::Int,
    L::Float64,
    beta::Float64,
    delta::Float64,
    p_swap::Float64,
    nsteps::Int,
    r_cut::Float64,
    dir_path::String;
    print_interval::Int=10000,
)
    # Initialize particles on a square lattice to avoid overlaps.
    particles = Particle[]
    (x, y) = initialize_lattice(N, L)
    for i in 1:N
        pos = SVector(x[i], y[i])
        sigma = inverse_diameters(; sigma_max=1.62)
        push!(particles, Particle(pos, sigma))
    end
    check_polidispersity(particles)

    # Determine the maximum sigma (and thus effective diameter) from the initial distribution.
    sigma_max_current = maximum(p.sigma for p in particles)
    # The maximum effective cutoff for any pair is r_cut * sigma_max.
    # Multiply by a small safety factor (e.g. 1.1) to ensure no neighbors are missed.
    effective_cutoff_for_cells = r_cut * sigma_max_current * 1.1

    # Build the initial cell list using the maximum effective cutoff.
    cell_list, particle_cells = build_cell_list(particles, L, effective_cutoff_for_cells)
    energies = zeros(Float64, nsteps)

    # Counters for acceptance ratios.
    disp_attempt = 0
    disp_accept = 0
    swap_attempt = 0
    swap_accept = 0

    # (Optionally, clear the energy log file at the start.)
    energy_file = joinpath(dir_path, "energy.txt")
    open(energy_file, "w") do io
        println(io, "# step energy")
    end

    for step in 1:nsteps
        move_type, accepted = mc_step!(
            particles, cell_list, particle_cells, beta, delta, p_swap, r_cut
        )
        if move_type == :disp
            disp_attempt += 1
            if accepted
                disp_accept += 1
            end
        elseif move_type == :swap
            swap_attempt += 1
            if accepted
                swap_accept += 1
            end
        end

        # Compute energy per particle.
        energies[step] = total_energy(particles, cell_list, r_cut) / N

        # Every 'print_interval' steps, print diagnostics, log energy, and write a snapshot.
        if step % print_interval == 0
            disp_ratio = (disp_attempt > 0) ? disp_accept / disp_attempt : 0.0
            swap_ratio = (swap_attempt > 0) ? swap_accept / swap_attempt : 0.0
            println(
                "Step $step: disp_acceptance = $(round(disp_ratio*100, digits=2))%, " *
                "swap_acceptance = $(round(swap_ratio*100, digits=2))%, Δ = $(round(delta, digits=4))",
            )
            # Write energy to file.
            open(energy_file, "a") do io
                println(io, "$step $(energies[step])")
            end
            # Write a snapshot in extended XYZ format.
            snapshot_filename = joinpath(dir_path, "snapshot_step_$(step).xyz")
            write_snapshot_xyz(snapshot_filename, particles, step, L)
            # Adjust Δ to approach a target acceptance ratio (here 30% as an example).
            delta = adjust_delta(delta, disp_accept, disp_attempt; target=0.3)
            # Reset counters for the next interval.
            disp_attempt = 0
            disp_accept = 0
            swap_attempt = 0
            swap_accept = 0
        end
    end
    return particles, energies
end

# ============================
# Main Program
# ============================
function main()
    # Parse command line arguments
    if length(ARGS) != 1
        println("Usage: julia monte_carlo_simulation.jl <temperature>")
        println("Example: julia monte_carlo_simulation.jl 0.05")
        exit(1)
    end

    temperature = parse(Float64, ARGS[1])

    # Simulation parameters.
    N = 20_000
    density = 1.0
    L = sqrt(N / density)
    beta = 1.0 / temperature         # Inverse temperature (1/kT).
    delta = 0.2                      # Initial maximum displacement.
    p_swap = 0.2                     # Probability for a swap move.
    nsteps = 1_000_000_000                 # Number of Monte Carlo steps.
    r_cut = 1.25                     # Base cutoff distance for the potential.

    # Use some of the parameters to make a special directory to
    # save the configurations
    save_dir = joinpath("large_system", "ktemp=$(temperature)_n=$(N)")
    mkpath(save_dir)

    # Run the simulation.
    _, energies = run_simulation(
        N, L, beta, delta, p_swap, nsteps, r_cut, save_dir; print_interval=1_000_000
    )
    println("Final energy: ", energies[end])

    return nothing
end

main()
