#pragma once

#include "Spin_Chain.hpp"
#include <cmath>
#include <random>


class Solver_Metropolis{
    public:
        void solve(Spin_chain &spin_data, double temp, std::mt19937& gen) const;
};


class SolverMetropolis2D {
public:
    void solve(SpinLattice2D& lattice, double temp, std::mt19937& gen) const;
    double calculate_delta_energy(const SpinLattice2D& lattice, 
                                size_t i, size_t j) const;
};