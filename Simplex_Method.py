import numpy as np
import pandas as pd

"""SIMPLEX METHOD CALCULATOR
   Author: Vladislav Zabrovsky"""

def create_simplex_input(num_variables, num_constraints):
    """Description:
       num_variables: how many variables are present
       num_constraints: how many constraint equetions are present
       Function serves for taking input of user,
       thus initializes A matrix,b vector and coefficients
       of objective function:
       Ax = b
       Max Z = c1x1 + .....cnxn - objective function for n variables
    """

    obj_coefficients = np.array(
        [float(input(f'Enter coefficient for variable x{i + 1} in the objective function: ')) for i in
         range(num_variables)])

    # Initialization of constraints parameters and parameters of b vector
    A_matrix_parameters = np.zeros((num_constraints, num_variables))
    B_matrix_parameters = np.zeros(num_constraints)

    for i in range(num_constraints):
        print(f'\nEnter coefficients for constraint {i + 1}:')
        for j in range(num_variables):
            A_matrix_parameters[i, j] = float(input(f' Coefficient for variable x{j + 1}: '))
        rhs_value = float(input('Enter the constraints from right side: '))
        if check_rhs_value(rhs_value):
            B_matrix_parameters[i] = rhs_value
        else:
            raise ValueError("Right hand side parameters must be >= 0 - condition for simplex method")
    return obj_coefficients, A_matrix_parameters, B_matrix_parameters


def print_linear_programming_problem(obj_coefficients, constraint_coefficients, rhs_values):
    """Description:
       Function serves for printing a linear problem
       to solve
       Parameters:
       obj_coefficients - coefficients of objective function
       constraint_coefficients - values of A matrix in Ax = b
       rhs_values - values of b vector in Ax = b

       Example:
           Objective Function:
           Maximize Z = 2.0x1 + -1.0x2

            System to solve:
            2.0x1 + 1.0x2 <= 18.0
            1.0x1 + 3.0x2 <= 12.0
            3.0x1 + -8.0x2 <= 16.0
            x1,x2 >= 0
    """
    num_variables = len(obj_coefficients)
    num_constraints = len(rhs_values)

    print("\nObjective Function:")
    objective_function = f"Maximize Z = {obj_coefficients[0]}x1"
    for i in range(1, num_variables):
        objective_function += f" + {obj_coefficients[i]}x{i + 1}"
    print(objective_function)

    print("\nSystem to solve:")
    for i in range(num_constraints):
        constraint = f"{constraint_coefficients[i, 0]}x1"
        for j in range(1, num_variables):
            constraint += f" + {constraint_coefficients[i, j]}x{j + 1}"
        constraint += f" <= {rhs_values[i]}"
        print(constraint)
    print(','.join(f"x{k + 1}" for k in range(num_variables)), end="")
    print(' >= 0')
    print(" ")


def create_simplex_table(obj_coefficients, constraint_coefficients, rhs_values):
    """Description:
       Function serves for creation of simplex table
       Parameters:
       obj_coefficients - coefficients of objective function
       constraint_coefficients - values of A matrix in Ax = b
       rhs_values - values of b vector in Ax = b
       """
    num_variables = len(obj_coefficients)
    num_constraints = len(rhs_values)

    # Create an identity matrix for the basis columns
    basis_addition = np.eye(num_constraints)

    # Augment the coefficients for the table
    augmented_matrix = np.hstack((rhs_values.reshape(-1, 1), constraint_coefficients, basis_addition))

    # Create Z_line for the objective function coefficients
    Z_line = np.hstack(
        (np.zeros(1), obj_coefficients * -1, np.zeros(augmented_matrix.shape[1] - obj_coefficients.shape[0] - 1)))

    # Stack the augmented_matrix and Z_line to form the tableau coefficients
    coefficients_for_table = np.vstack((augmented_matrix, Z_line))

    # Creation of pandas DataFrame for data storage
    df = pd.DataFrame(coefficients_for_table,
                      columns=['b'] + [f'x{i + 1}' for i in range(num_variables + num_constraints)])

    # Set 'Basis' as the index
    df.insert(0, 'Basis', [f'x{i + num_variables + 1}' for i in range(num_constraints)] + ['Z'])
    df.set_index("Basis", inplace=True)

    return df


def print_simplex_table(simplex_table):
    """Description:
       Prints Simplex tables"""
    print("SIMPLEX METHOD TABLE")
    print(simplex_table)


def perform_gaussian_elimination(df, pivot_row, pivot_column):
    """Jordan Gauss method for recalculation of coefficients"""
    for basis in df.index:
        if basis != pivot_row:
            multiplier = df.at[basis, pivot_column]
            df.loc[basis] -= multiplier * df.loc[pivot_row]
    return df


def simplex_method(obj_coefficients, constraint_coefficients, rhs_values):
    """Description:
       Basic implementation of simplex method algorithm
       """
    df = create_simplex_table(obj_coefficients, constraint_coefficients, rhs_values)

    iterations = 0

    while np.min(df.loc["Z"][1:]) < 0:
        valid_columns = df.columns[1:]  # we don't include b column
        pivot_column = df.loc["Z", valid_columns].idxmin()
        ratio = df['b'].drop('Z') / df[pivot_column].drop('Z')
        ratio = ratio[ratio > 0]
        pivot_row = ratio.idxmin()

        main_element = df.at[pivot_row, pivot_column]
        df.loc[pivot_row] /= main_element
        df = perform_gaussian_elimination(df, pivot_row, pivot_column)
        df = df.rename(index={pivot_row: pivot_column})

        iterations += 1
        print(f"Iteration: {iterations}")
        print_simplex_table(df)
        print(f"Pivot_column: {pivot_column} ")
        print(f"Pivot_row: {pivot_row} ")
        print(f"Main_element: {main_element}")
        print(" " * 20)
    optimal_z = df.at['Z', 'b']

    main_variables = [f'x{i}' for i in range(1, len(obj_coefficients) + 1)]

    # Initialize a dictionary with default values of 0,we will use it for writing an optimized vector
    optimal_vector = {var: 0 for var in main_variables}

    # Update the dictionary with actual values
    for var in df.index:
        if var in optimal_vector:
            optimal_vector[var] = df.at[var, 'b']

    print(f"Optimal Z value: {optimal_z}")
    print(f"Optimal Vector: {optimal_vector}")


def check_rhs_value(user_input):
    if user_input >= 0:
        return True
    return False

