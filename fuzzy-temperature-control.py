import numpy as np
import matplotlib.pyplot as plt

# Fuzzy sets
temperature = {'cold': [0, 10, 20],
               'moderate': [15, 25, 35],
               'hot': [30, 40, 50]}

heating_power = {'low': [0, 20, 40],
                 'medium': [30, 50, 70],
                 'high': [60, 80, 100]}

# Fuzzy rules
rules = {'rule1': {'temperature': 'cold', 'heating_power': 'high'},
         'rule2': {'temperature': 'moderate', 'heating_power': 'medium'},
         'rule3': {'temperature': 'hot', 'heating_power': 'low'}}



# Fuzzification
def fuzzify(x, fuzzy_set):
    membership = {}
    for key, value in fuzzy_set.items():
        # Fix: Use [0, 1, 0] as fp to match the length of value (xp)
        membership[key] = np.interp(x, value, [0, 1, 0])
    return membership

def defuzzify(membership, fuzzy_set):
    numerator = 0
    denominator = 0

    for key, value in membership.items():
        crisp_set = fuzzy_set[key]
        centroid = np.mean(crisp_set)
        numerator += centroid * value
        denominator += value

    if denominator != 0:
        crisp_output = numerator / denominator
    else:
        crisp_output = 0

    return crisp_output

# Temperature input
input_temp = 25

# Fuzzification of input
temp_membership = fuzzify(input_temp, temperature)

# Apply fuzzy rules and compute firing strength
firing_strength = {}
for rule, values in rules.items():
    temp_mem = temp_membership[values['temperature']]
    firing_strength[rule] = temp_mem

# Compute output membership
output_membership = {}
for rule, values in rules.items():
    output_membership[values['heating_power']] = np.minimum(output_membership.get(values['heating_power'], 0), firing_strength[rule])

# Defuzzify output
output_power = defuzzify(output_membership, heating_power)

# Print the crisp output
print('Input Temperature:', input_temp)
print('Output Heating Power:', output_power)

# Plot membership functions
x = np.linspace(0, 50, 100)

plt.figure(figsize=(10, 6))

for key, value in temperature.items():
    membership = fuzzify(x, {key: value})
    # Fix: Remove the extra list by directly accessing membership[key]
    plt.plot(x, membership[key], label=key)

plt.xlabel('Temperature')
plt.ylabel('Membership')
plt.title('Temperature Membership Functions')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))

for key, value in heating_power.items():
    membership = fuzzify(x, {key: value})
    # Fix: Access the membership value directly instead of using list comprehension
    plt.plot(x, membership[key], label=key)

plt.xlabel('Heating Power')
plt.ylabel('Membership')
plt.title('Heating Power Membership Functions')
plt.legend()
plt.grid(True)
plt.show()
