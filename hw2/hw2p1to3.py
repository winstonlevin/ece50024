import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv

# Exercise 1 ===========================================================================================================
# Reading csv file for male data
with open("male_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    male_data_list = list(reader)
    male_data_types = male_data_list[:][0]
    male_data_arr = np.array(male_data_list[:][1:], dtype=float)
    male_bmi = male_data_arr[:, 1]
    male_stature_mm = male_data_arr[:, 2]
    male_bmi_normalized = male_bmi * 1e-1
    male_stature_normalized = male_stature_mm * 1e-3
csv_file.close()

# Reading csv file for female data
with open("female_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    female_data_list = list(reader)
    female_data_types = female_data_list[:][0]
    female_data_arr = np.array(female_data_list[:][1:], dtype=float)
    female_bmi = female_data_arr[:, 1]
    female_stature_mm = female_data_arr[:, 2]
    female_bmi_normalized = female_bmi * 1e-1
    female_stature_normalized = female_stature_mm * 1e-3
csv_file.close()

# Print data
print('| (idx) | Female BMI (norm.) | Female Stature (norm.) | Male BMI (norm.) | Male Stature (norm.) |')
print('|_______________________________________________________________________________________________|')

for idx, (f_bmi, f_stat, m_bmi, m_stat) in enumerate(zip(
        female_bmi_normalized, female_stature_normalized, male_bmi_normalized, male_stature_normalized
)):
    print(f'|   {idx:<3d} | {f_bmi:<18.4f} | {f_stat:<22.4f} | {m_bmi:<16.4f} | {m_stat:<20.4f} |')
    if idx == 9:
        break

# Exercise 2 ===========================================================================================================
# Exercise 2 (b) -------------------------------------------------------------------------------------------------------

# Exercise 2 (c) -------------------------------------------------------------------------------------------------------

# Exercise 2 (e) -------------------------------------------------------------------------------------------------------

# Exercise 2 (f) -------------------------------------------------------------------------------------------------------

# Exercise 2 (g) -------------------------------------------------------------------------------------------------------

# Exercise 2 (h) -------------------------------------------------------------------------------------------------------

# Exercise 3 ===========================================================================================================
# Exercise 3 (a) -------------------------------------------------------------------------------------------------------

# Exercise 3 (b) -------------------------------------------------------------------------------------------------------
