# spin_chain.pyx
import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "include/Library.hpp":
    void spin_1d(double* data_spins,
                 int initial_condition,
                 int spin_number,
                 double temp,
                 double spin_j,
                 double b_field,
                 double bohr_magneton,
                 int iteration)

    double chain_energy(double* data_spins,
                       int spin_number,
                       double temp,
                       double spin_j,
                       double b_field,
                       double bohr_magneton)

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
    
    void average_magnetization(double* result_buffer,
                        int initial_condition,
                        int iteration,
                        double temp_min,
                        double temp_max,
                        double dt,
                        double spin_j,
                        double b_field,
                        int spin_number,
                        double bohr_magneton)
    
    void average_heat_capacity(double* result_buffer,
                        int initial_condition,
                        int iteration,
                        double temp_min,
                        double temp_max,
                        double dt,
                        double spin_j,
                        double b_field,
                        int spin_number,
                        double bohr_magneton)

    void simulate_2d(double* data_spins,
                    int initial_condition,
                    int lattice_size,
                    double temp,
                    double spin_j,
                    double b_field,
                    double bohr_magneton,
                    int iteration)
    
    double calculate_lattice_energy(double* spin_data,
                                  int lattice_size,
                                  double temp,
                                  double spin_j,
                                  double b_field,
                                  double bohr_magneton)
    
    void calculate_average_energy_2d(double* result_buffer,
                                 int initial_condition,
                                 int iterations,
                                 double temp_min,
                                 double temp_max,
                                 double dt,
                                 double spin_j,
                                 double b_field,
                                 int lattice_size,
                                 double bohr_magneton)
    
    void calculate_average_magnetization_2d(double* result_buffer,
                                       int initial_condition,
                                       int iterations,
                                       double temp_min,
                                       double temp_max,
                                       double dt,
                                       double spin_j,
                                       double b_field,
                                       int lattice_size,
                                       double bohr_magneton)
    
    void calculate_heat_capacity_2d(double* result_buffer,
                               int initial_condition,
                               int iterations,
                               double temp_min,
                               double temp_max,
                               double dt,
                               double spin_j,
                               double b_field,
                               int lattice_size,
                               double bohr_magneton)

def spin_chain_1d(initial_condition: int, spin_number: int,
                  temp: float, spin_j: float, b_field: float,
                  bohr_magneton: float, iteration: int):

    cdef cnp.ndarray[double, ndim=2] spins = np.zeros((iteration, spin_number), dtype=np.float64)
    cdef double[:, :] spins_view = spins

    spin_1d(&spins_view[0,0], initial_condition, spin_number, 
            temp, spin_j, b_field, bohr_magneton, iteration)
    return spins

def chain_energy_py(data_spin, spin_number: int, temp: float, 
                    spin_j: float, b_field: float, bohr_magneton: float):

    cdef cnp.ndarray[double, ndim=1] arr = np.asarray(data_spin, dtype=np.float64)
    cdef double[:] arr_view = arr

    return chain_energy(&arr_view[0], spin_number, temp, spin_j, b_field, bohr_magneton)

def average_energy_py(iteration: int, initial_condition: int, temp_min: float, temp_max: float, dt: float,
                      spin_j: float, b_field: float, spin_number: int,
                      bohr_magneton: float):
    cdef int temp_points = int((temp_max - temp_min) / dt) 
    cdef cnp.ndarray[double, ndim=1] result = np.empty(temp_points, dtype=np.float64)
    cdef double[:] result_view = result
    average_energy(&result_view[0], initial_condition, iteration, temp_min, temp_max, dt,
                   spin_j, b_field, spin_number, bohr_magneton)
    return result

def average_magnetization_py(iterations: int, initial_condition: int, temp_min: float, temp_max: float, dt: float,
                             spin_j: float, b_field: float, spin_number: int,
                             bohr_magneton: float):
    cdef int temp_points = int((temp_max - temp_min) / dt) 
    cdef cnp.ndarray[double, ndim=1] result = np.empty(temp_points, dtype=np.float64)
    cdef double[:] result_view = result
    average_magnetization(&result_view[0], initial_condition, iterations, temp_min, temp_max, dt,
                          spin_j, b_field, spin_number, bohr_magneton)
    return result

def average_heat_capacity_py(iterations: int, initial_condition: int, temp_min: float, temp_max: float, dt: float,
                             spin_j: float, b_field: float, spin_number: int,
                             bohr_magneton: float):
    cdef int temp_points = int((temp_max - temp_min) / dt) 
    cdef cnp.ndarray[double, ndim=1] result = np.empty(temp_points, dtype=np.float64)
    cdef double[:] result_view = result
    average_heat_capacity(&result_view[0], initial_condition, iterations, temp_min, temp_max, dt,
                          spin_j, b_field, spin_number, bohr_magneton)
    return result


def simulate_2d_py(initial_condition: int, lattice_size: int,
                   temp: float, spin_j: float, b_field: float,
                   bohr_magneton: float, iteration: int):
    cdef int total_spins = lattice_size * lattice_size
    cdef cnp.ndarray[double, ndim=3] spins = np.zeros((iteration, lattice_size, lattice_size), dtype=np.float64)
    cdef double[:, :, :] spins_view = spins
    
    simulate_2d(&spins_view[0,0,0], initial_condition, lattice_size, 
               temp, spin_j, b_field, bohr_magneton, iteration)
    return spins

def lattice_energy_py(spin_config, lattice_size: int, temp: float, 
                     spin_j: float, b_field: float, bohr_magneton: float):
    cdef cnp.ndarray[double, ndim=2] arr = np.asarray(spin_config, dtype=np.float64)
    cdef double[:, :] arr_view = arr
    return calculate_lattice_energy(&arr_view[0,0], lattice_size, temp, spin_j, b_field, bohr_magneton)

def average_energy_2d_py(iterations: int, initial_condition: int, 
                        temp_min: float, temp_max: float, dt: float,
                        spin_j: float, b_field: float, lattice_size: int,
                        bohr_magneton: float):
    cdef int temp_points = int((temp_max - temp_min) / dt) + 1
    cdef cnp.ndarray[double, ndim=1] result = np.empty(temp_points, dtype=np.float64)
    cdef double[:] result_view = result
    
    calculate_average_energy_2d(&result_view[0], initial_condition, iterations,
                           temp_min, temp_max, dt, spin_j, b_field,
                           lattice_size, bohr_magneton)
    return result

def average_magnetization_2d_py(iterations: int, initial_condition: int,
                              temp_min: float, temp_max: float, dt: float,
                              spin_j: float, b_field: float, lattice_size: int,
                              bohr_magneton: float):
    cdef int temp_points = int((temp_max - temp_min) / dt) + 1
    cdef cnp.ndarray[double, ndim=1] result = np.empty(temp_points, dtype=np.float64)
    cdef double[:] result_view = result
    
    calculate_average_magnetization_2d(&result_view[0], initial_condition, iterations,
                                  temp_min, temp_max, dt, spin_j, b_field,
                                  lattice_size, bohr_magneton)
    return result

def average_heat_capacity_2d_py(iterations: int, initial_condition: int,
                               temp_min: float, temp_max: float, dt: float,
                               spin_j: float, b_field: float, lattice_size: int,
                               bohr_magneton: float):
    cdef int temp_points = int((temp_max - temp_min) / dt) + 1
    cdef cnp.ndarray[double, ndim=1] result = np.empty(temp_points, dtype=np.float64)
    cdef double[:] result_view = result
    
    calculate_heat_capacity_2d(&result_view[0], initial_condition, iterations,
                          temp_min, temp_max, dt, spin_j, b_field,
                          lattice_size, bohr_magneton)
    return result