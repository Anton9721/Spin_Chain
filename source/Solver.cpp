#include "Solver.hpp"
#include <random>
#include <cmath>

void Solver_Metropolis::solve(Spin_chain &spin_data, double temp, std::mt19937& gen) const
{
    size_t change_spin = spin_data.get_random_spin(gen);  

    std::vector<double> s = spin_data.get_spins(); 
    s[change_spin] *= -1;  

    double energy_old = spin_data.calculate_energy();  
    spin_data.set_spins(s);  
    double energy_new = spin_data.calculate_energy();


    if (energy_new > energy_old) { 
        std::uniform_real_distribution<double> dist(0.0, 1.0); 
        double random_value = dist(gen);

        double r_probability = exp(- (energy_new - energy_old) / temp);
        if (r_probability < random_value) {
            s[change_spin] *= -1; 
            spin_data.set_spins(s);
        }
    }
}

// -------------------------------------------------------------------------------------

void SolverMetropolis2D::solve(SpinLattice2D& lattice, double temp, std::mt19937& gen) const {
    auto [i, j] = lattice.get_random_spin(gen);
    const size_t n = lattice.get_size();
    const size_t idx = i * n + j;
    
    std::vector<double>& spins = lattice.get_spins_mutable();
    
    double delta_E = -2.0 * calculate_delta_energy(lattice, i, j);
    
    if (delta_E <= 0) {
        spins[idx] *= -1;  
    } else {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(gen) < exp(-delta_E / temp)) {
            spins[idx] *= -1;  
        }
    }
}

double SolverMetropolis2D::calculate_delta_energy(const SpinLattice2D& lattice, 
                                                size_t i, size_t j) const {
    const size_t n = lattice.get_size();
    const size_t idx = i * n + j;
    const double spin = lattice.get_spins()[idx];
    double energy_contribution = 0.0;
    
    energy_contribution += lattice.get_spins()[i * n + (j + 1) % n];      
    energy_contribution += lattice.get_spins()[i * n + (j - 1 + n) % n];  
    energy_contribution += lattice.get_spins()[((i + 1) % n) * n + j];    
    energy_contribution += lattice.get_spins()[((i - 1 + n) % n) * n + j];
    
    return -lattice.get_spin_j() * spin * energy_contribution 
           - lattice.get_b_field() * lattice.get_bohr_magneton() * spin;
}