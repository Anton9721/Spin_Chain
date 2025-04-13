#pragma once

#include "Spin_Chain.hpp"
#include "Solver.hpp"
#include <span>
#include <stdexcept>
#include <vector>

extern "C" {
    void spin_1d(double* data_spins,
                 int initial_condition,
                 int spin_number,
                 double temp,
                 double spin_j,
                 double b_field,
                 double bohr_magneton,
                 int iteration);

    double chain_energy(double* data_spins,
                        int spin_number,
                        double temp,
                        double spin_j,
                        double b_field,
                        double bohr_magneton);

    void average_energy(double* result_buffer,
                        int initial_condition,
                        int iterations,
                        double temp_min,
                        double temp_max,
                        double dt,
                        double spin_j,
                        double b_field,
                        int spin_number,
                        double bohr_magneton);
    
    void average_magnetization(double* result_buffer,
                        int initial_condition,
                        int iterations,
                        double temp_min,
                        double temp_max,
                        double dt,
                        double spin_j,
                        double b_field,
                        int spin_number,
                        double bohr_magneton);
    void average_heat_capacity(double* result_buffer,
                        int initial_condition,
                        int iterations,
                        double temp_min,
                        double temp_max,
                        double dt,
                        double spin_j,
                        double b_field,
                        int spin_number,
                        double bohr_magneton);

    void simulate_2d(double* output_data, 
                            int initial_condition,
                            int lattice_size,
                            double temp,
                            double spin_j,
                            double b_field,
                            double bohr_magneton,
                            int iterations);

    double calculate_lattice_energy(double* spin_data,
                              int lattice_size,
                              double temp,
                              double spin_j,
                              double b_field,
                              double bohr_magneton);

    void calculate_average_energy_2d(double* result_buffer,
                                        int initial_condition,
                                        int iterations,
                                        double temp_min,
                                        double temp_max,
                                        double dt,
                                        double spin_j,
                                        double b_field,
                                        int lattice_size,
                                        double bohr_magneton);

    void calculate_average_magnetization_2d(double* result_buffer,
                                                int initial_condition,
                                                int iterations,
                                                double temp_min,
                                                double temp_max,
                                                double dt,
                                                double spin_j,
                                                double b_field,
                                                int lattice_size,
                                                double bohr_magneton);

    void calculate_heat_capacity_2d(double* result_buffer,
                                        int initial_condition,
                                        int iterations,
                                        double temp_min,
                                        double temp_max,
                                        double dt,
                                        double spin_j,
                                        double b_field,
                                        int lattice_size,
                                        double bohr_magneton);

}