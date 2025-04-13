#include "Spin_Chain.hpp"

Spin_chain::Spin_chain(size_t n, double j, double b, double magneton, double temp, std::mt19937& gen)
    : spin_number(n), spin_j(j), b_field(b), bohr_magneton(magneton), temperature(temp), spin_data(n) {

    std::uniform_int_distribution<int> dist(0, 1);
    
    for (size_t i = 0; i < spin_number; ++i) {
        spin_data[i] = dist(gen) ? 1 : -1;
    }
}

void Spin_chain::spin_chain_cold() {
    for (size_t i = 0; i < spin_number; ++i) {
        spin_data[i] = static_cast<double>(-1); 
    }
}


double Spin_chain::calculate_energy() const {
    double energy = 0.0;
    for (size_t i = 0; i < spin_number; ++i) {
        size_t next = (i + 1) % spin_number;
        energy += -spin_j * spin_data[i] * spin_data[next] - b_field * bohr_magneton * spin_data[i];
    }
    return energy;
}


const std::vector<double>& Spin_chain::get_spins() const {
    return spin_data;
}

size_t Spin_chain::get_random_spin(std::mt19937& gen) const {
    std::uniform_int_distribution<size_t> dist(0, spin_number - 1); 
    return dist(gen);  
}

void Spin_chain::set_spins(const std::vector<double>& new_spins) {
    spin_data = new_spins;
}


// ---------------------------------------------------------------------------------------------------

SpinLattice2D::SpinLattice2D(size_t size, double j, double b, double magneton, double temp, std::mt19937& gen)
    : lattice_size(size), spin_j(j), b_field(b), bohr_magneton(magneton), 
      temperature(temp), spin_data(size * size) {

    std::uniform_int_distribution<int> dist(0, 1);
    for (auto& spin : spin_data) {
        spin = dist(gen) ? 1.0 : -1.0;
    }
}

void SpinLattice2D::initialize_cold() {
    std::fill(spin_data.begin(), spin_data.end(), -1.0);
}

double SpinLattice2D::calculate_energy() const {
    double energy = 0.0;
    const size_t n = lattice_size;
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const size_t idx = i * n + j;
            const double current_spin = spin_data[idx];
            
            const size_t right = i * n + (j + 1) % n;
            energy += -spin_j * current_spin * spin_data[right];
            
            const size_t down = ((i + 1) % n) * n + j;
            energy += -spin_j * current_spin * spin_data[down];
            
            energy += -b_field * bohr_magneton * current_spin;
        }
    }
    return energy;
}

const std::vector<double>& SpinLattice2D::get_spins() const {
    return spin_data;
}

std::pair<size_t, size_t> SpinLattice2D::get_random_spin(std::mt19937& gen) const {
    std::uniform_int_distribution<size_t> dist(0, lattice_size - 1);
    return {dist(gen), dist(gen)};
}

void SpinLattice2D::set_spins(const std::vector<double>& new_spins) {
    spin_data = new_spins;
}

std::vector<double>& SpinLattice2D::get_spins_mutable() {
    return spin_data;
}
