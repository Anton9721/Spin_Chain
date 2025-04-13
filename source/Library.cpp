#include "Library.hpp"


namespace {
    std::random_device rd;
    std::mt19937 gen(rd());
}


void spin_1d(double* data_spins,
             int initial_condition,
             int spin_number,
             double temp,
             double spin_j,
             double b_field,
             double bohr_magneton,
             int iteration)
{
    std::span<double> output(data_spins, spin_number * iteration);
    Spin_chain chain(spin_number, spin_j, b_field, bohr_magneton, temp, gen);
    Solver_Metropolis solver;

    if (initial_condition == 1) { // 1 для "cold"
        chain.spin_chain_cold();
    }

    for (int i = 0; i < iteration; ++i) {
        solver.solve(chain, temp, gen);
        const auto& spins = chain.get_spins();
        std::copy(spins.begin(), spins.end(), output.begin() + i * spin_number);
    }
}

double chain_energy(double* data_spins,
                    int spin_number,
                    double temp,
                    double spin_j,
                    double b_field,
                    double bohr_magneton)
{
    std::span<double> spins(data_spins, spin_number);
    Spin_chain chain(spin_number, spin_j, b_field, bohr_magneton, temp, gen);
    chain.set_spins(std::vector<double>(spins.begin(), spins.end()));
    return chain.calculate_energy();
}

void average_energy(double* result_buffer,
                   int initial_condition,
                   int iteration,
                   double temp_min,
                   double temp_max,
                   double dt,
                   double spin_j,
                   double b_field,
                   int spin_number,
                   double bohr_magneton)
{
    const int warmup_iterations = 15 * spin_number;
    
    const int temp_points = static_cast<int>((temp_max - temp_min) / dt);
    std::span<double> result(result_buffer, temp_points);

    #pragma omp parallel for
    for (int i = 0; i < temp_points; ++i) {
        double temp = temp_min + i * dt;
        double energy_sum = 0.0;
        int count = 0;

        Spin_chain chain(spin_number, spin_j, b_field, bohr_magneton, temp, gen);
        Solver_Metropolis solver;
        if (initial_condition == 1) { // 1 для "cold"
            chain.spin_chain_cold();
        }

        for (int j = 0; j < warmup_iterations; ++j) {
            solver.solve(chain, temp, gen);
        }

        for (int j = warmup_iterations; j < iteration; ++j) {
            solver.solve(chain, temp, gen);
            energy_sum += chain.calculate_energy();
            count++;
        }

        result[i] = energy_sum / (count * spin_number);
    }
}

void average_magnetization(double* result_buffer,
                            int initial_condition,
                            int iterations,
                            double temp_min,
                            double temp_max,
                            double dt,
                            double spin_j,
                            double b_field,
                            int spin_number,
                            double bohr_magneton)
{
    const int warmup_iterations = 15 * spin_number;

    const int temp_points = static_cast<int>((temp_max - temp_min) / dt);
    std::span<double> result(result_buffer, temp_points);

    #pragma omp parallel for
    for (int i = 0; i < temp_points; ++i) {
        double temp = temp_min + i * dt;
        double mag_sum = 0.0;
        int count = 0;

        Spin_chain chain(spin_number, spin_j, b_field, bohr_magneton, temp, gen);
        Solver_Metropolis solver;
        if (initial_condition == 1) { // 1 для "cold"
            chain.spin_chain_cold();
        }

        for (int j = 0; j < warmup_iterations; ++j) {
            solver.solve(chain, temp, gen);
        }

        for (int j = warmup_iterations; j < iterations; ++j) {
            solver.solve(chain, temp, gen);
            const auto& spins = chain.get_spins();
            double sum = 0.0;
            for (double s : spins) sum += s;
            mag_sum += sum / spin_number;
            count++;
        }

        result[i] = mag_sum / count;
    }
}

void average_heat_capacity(double* result_buffer,
                        int initial_condition,
                        int iterations,
                        double temp_min,
                        double temp_max,
                        double dt,
                        double spin_j,
                        double b_field,
                        int spin_number,
                        double bohr_magneton)
{
    const int warmup_iterations = 15 * spin_number;

    const int temp_points = static_cast<int>((temp_max - temp_min) / dt);
    std::span<double> result(result_buffer, temp_points);

    #pragma omp parallel for
    for (int i = 0; i < temp_points; ++i) {
        double temp = temp_min + i * dt;
        double energy_sum = 0.0;
        double energy_sq_sum = 0.0;
        int count = 0;

        Spin_chain chain(spin_number, spin_j, b_field, bohr_magneton, temp, gen);
        Solver_Metropolis solver;
        if (initial_condition == 1) { // 1 для "cold"
            chain.spin_chain_cold();
        }

        for (int j = 0; j < warmup_iterations; ++j) {
            solver.solve(chain, temp, gen);
        }

        for (int j = warmup_iterations; j < iterations; ++j) {
            solver.solve(chain, temp, gen);
            double e = chain.calculate_energy();
            energy_sum += e;
            energy_sq_sum += e * e;
            count++;
        }

        double avg_e = energy_sum / count;
        double avg_e_sq = energy_sq_sum / count;

        if(temp < 1.0){
            result[i] = (avg_e_sq - avg_e * avg_e) / (temp * spin_number);
        }
        else{
            result[i] = (avg_e_sq - avg_e * avg_e) / (temp * temp * spin_number);
        }
    }
}

// ----------------------------------------------------------------




void simulate_2d(double* output_data, 
                int initial_condition,
                int lattice_size,
                double temp,
                double spin_j,
                double b_field,
                double bohr_magneton,
                int iterations) {
    const int total_spins = lattice_size * lattice_size;
    std::span<double> output(output_data, total_spins * iterations);
    
    SpinLattice2D lattice(lattice_size, spin_j, b_field, bohr_magneton, temp, gen);
    SolverMetropolis2D solver;

    if (initial_condition == 1) {
        lattice.initialize_cold();
    }

    const auto& initial_spins = lattice.get_spins();
    std::copy(initial_spins.begin(), initial_spins.end(), output.begin());

    for (int i = 1; i < iterations; ++i) {
        solver.solve(lattice, temp, gen);
        const auto& current_spins = lattice.get_spins();
        std::copy(current_spins.begin(), current_spins.end(), 
                 output.begin() + i * total_spins);
    }
}

double calculate_lattice_energy(double* spin_data,
                                int lattice_size,
                                double temp,
                                double spin_j,
                                double b_field,
                                double bohr_magneton) {
    std::vector<double> spins(spin_data, spin_data + lattice_size * lattice_size);
    SpinLattice2D lattice(lattice_size, spin_j, b_field, bohr_magneton, temp, gen);
    lattice.set_spins(spins);
    return lattice.calculate_energy() / (lattice_size * lattice_size); // Нормировка на спин
}

void calculate_average_energy_2d(double* result_buffer,
                                int initial_condition,
                                int iterations,
                                double temp_min,
                                double temp_max,
                                double dt,
                                double spin_j,
                                double b_field,
                                int lattice_size,
                                double bohr_magneton) {
    const int warmup = 15 * lattice_size * lattice_size;
    const int temp_points = static_cast<int>((temp_max - temp_min) / dt) + 1;
    std::span<double> result(result_buffer, temp_points);

    #pragma omp parallel for
    for (int i = 0; i < temp_points; ++i) {
        double temp = temp_min + i * dt;
        double energy_sum = 0.0;
        int count = 0;

        SpinLattice2D lattice(lattice_size, spin_j, b_field, bohr_magneton, temp, gen);
        SolverMetropolis2D solver;
        
        if (initial_condition == 1) {
            lattice.initialize_cold();
        }

        for (int j = 0; j < warmup; ++j) {
            solver.solve(lattice, temp, gen);
        }

        for (int j = warmup; j < iterations; ++j) {
            solver.solve(lattice, temp, gen);
            energy_sum += lattice.calculate_energy();
            count++;
        }

        result[i] = energy_sum / (count * lattice_size * lattice_size);
    }
}


void calculate_average_magnetization_2d(double* result_buffer,
                                                     int initial_condition,
                                                     int iterations,
                                                     double temp_min,
                                                     double temp_max,
                                                     double dt,
                                                     double spin_j,
                                                     double b_field,
                                                     int lattice_size,
                                                     double bohr_magneton) {
    const int warmup = 15 * lattice_size * lattice_size;
    const int temp_points = static_cast<int>((temp_max - temp_min) / dt) + 1;
    std::span<double> result(result_buffer, temp_points);

    #pragma omp parallel for
    for (int i = 0; i < temp_points; ++i) {
        double temp = temp_min + i * dt;
        double magnetization_sum = 0.0;
        int count = 0;

        SpinLattice2D lattice(lattice_size, spin_j, b_field, bohr_magneton, temp, gen);
        SolverMetropolis2D solver;
        
        if (initial_condition == 1) {
            lattice.initialize_cold();
        }

        for (int j = 0; j < warmup; ++j) {
            solver.solve(lattice, temp, gen);
        }

        for (int j = warmup; j < iterations; ++j) {
            solver.solve(lattice, temp, gen);
            
            const auto& spins = lattice.get_spins();
            double mag = 0.0;
            for (double s : spins) {
                mag += s;
            }
            magnetization_sum += std::abs(mag) / (lattice_size * lattice_size); // Абсолютное значение и нормировка
            count++;
        }

        result[i] = magnetization_sum / count;
    }
}

void calculate_heat_capacity_2d(double* result_buffer,
                                             int initial_condition,
                                             int iterations,
                                             double temp_min,
                                             double temp_max,
                                             double dt,
                                             double spin_j,
                                             double b_field,
                                             int lattice_size,
                                             double bohr_magneton) {
    const int warmup = 15 * lattice_size * lattice_size;
    const int temp_points = static_cast<int>((temp_max - temp_min) / dt) + 1;
    std::span<double> result(result_buffer, temp_points);

    #pragma omp parallel for
    for (int i = 0; i < temp_points; ++i) {
        double temp = temp_min + i * dt;
        double energy_sum = 0.0;
        double energy_sq_sum = 0.0;
        int count = 0;

        SpinLattice2D lattice(lattice_size, spin_j, b_field, bohr_magneton, temp, gen);
        SolverMetropolis2D solver;
        
        if (initial_condition == 1) {
            lattice.initialize_cold();
        }

        for (int j = 0; j < warmup; ++j) {
            solver.solve(lattice, temp, gen);
        }

        for (int j = warmup; j < iterations; ++j) {
            solver.solve(lattice, temp, gen);
            double e = lattice.calculate_energy();
            energy_sum += e;
            energy_sq_sum += e * e;
            count++;
        }

        double avg_e = energy_sum / count;
        double avg_e_sq = energy_sq_sum / count;
        double variance = avg_e_sq - avg_e * avg_e;
        
        result[i] = variance / (temp * temp * lattice_size * lattice_size);
    }
}