from Simplex_Method import print_simplex_table,print_linear_programming_problem,create_simplex_table,create_simplex_input,simplex_method

def main():
    num_variables = int(input("Enter the number of variables: "))
    if num_variables <= 1:
        raise ValueError("Number of variables should be at least 2")
    num_constraints = int(input("Enter the number of constraints: "))
    if num_constraints <= 0:
        raise ValueError("Invalid quantity of constraints")

    obj_coefficients, constraint_coefficients, rhs_values = create_simplex_input(num_variables, num_constraints)
    print_linear_programming_problem(obj_coefficients, constraint_coefficients, rhs_values)

    table = create_simplex_table(obj_coefficients, constraint_coefficients, rhs_values)
    print_simplex_table(table)

    simplex_method(obj_coefficients, constraint_coefficients, rhs_values)


if __name__ == "__main__":
    try:
        print("""

SIMPLEX METHOD CALCULATOR
Author: Vladislav Zabrovsky

        """)
        main()
    except Exception as e:
        print(f"Wow,the problem appeared.Explanation: {e}")
