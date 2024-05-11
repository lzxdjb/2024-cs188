from itertools import permutations

# Sample list
original_list = [1, 2, 3]

original_list = list(range(1, 4))

# Generate permutations
permutation_list = list(permutations(original_list))

# Print the result
print("Original List:", original_list)
print("Permutations:", permutation_list)
