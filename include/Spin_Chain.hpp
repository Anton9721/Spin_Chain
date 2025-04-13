
#pragma once
#include <vector>
#include <random>

class Spin_chain {
private:
    size_t spin_number;
    double spin_j;
    double b_field;
    double bohr_magneton;
    double temperature;
    std::vector<double> spin_data;

public:
    Spin_chain(size_t n, double j, double b, double magneton, double temp, std::mt19937& gen);
    void spin_chain_cold();
    double calculate_energy() const;
    const std::vector<double>& get_spins() const;
    size_t get_random_spin(std::mt19937& gen) const;
    void set_spins(const std::vector<double>& new_spins);
};


class SpinLattice2D {
private:
    size_t lattice_size;
    double spin_j;
    double b_field;
    double bohr_magneton;
    double temperature;
    std::vector<double> spin_data;

public:
    SpinLattice2D(size_t size, double j, double b, double magneton, double temp, std::mt19937& gen);
    
    double get_spin_j() const { return spin_j; }
    double get_b_field() const { return b_field; }
    double get_bohr_magneton() const { return bohr_magneton; }
    double get_temperature() const { return temperature; }
    size_t get_size() const { return lattice_size; }
    
    void initialize_cold();
    double calculate_energy() const;
    const std::vector<double>& get_spins() const;
    std::vector<double>& get_spins_mutable(); 
    std::pair<size_t, size_t> get_random_spin(std::mt19937& gen) const;
    void set_spins(const std::vector<double>& new_spins);
};