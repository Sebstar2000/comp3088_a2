import math
from re import A
import pandas as pd
import numpy as np
import numpy.ma as ma

def get_file_df(filename, col_names, test=False):
    headers = ''
    new_col = col_names.copy()
    if test:
        new_col = col_names[:-1]
    df = pd.read_csv(filename, names=new_col, index_col=False)
    return df

def get_file_data(filename, test):
    arr = []
    solution = []
    
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            line = line.strip()
            line = line.split(',')
            if not test:
                solution.append(line[-1])
                line = line[:-1]
            line = [float(i) for i in line]
            arr.append(line)
            
    return arr, solution

def gen_eucl_dist(x, A):
    new_matrix = (A - x)**2 ## Minus the cur_test from each row in train and square each value
    vector_sums = new_matrix @ np.ones(len(x)) # Get the Matrix multiplication calculating the vector of pre rooted sums
    final_dist = np.sqrt(vector_sums) # Square root each value in the vector
    return final_dist

def classify_nn(training_filename, testing_filename, k, overide_data=False):
    col_names = ['num_preg','gluc_conc','bld_prs','skn_thck','insulin','bmi','dia_ped_f','age','class']
    nn_classific_cls = []
    
    # Taking data from a file, or provided for 10-fold 
    if overide_data == False:
        train_df, solutions = get_file_data(training_filename, False)
        test_df, s = get_file_data(testing_filename, True)
    else:
        train_df = training_filename[0]
        solutions = training_filename[1]
        test_df = testing_filename
    
    # print(train_df[:5])
    # print(solutions[:5])
    # print(test_df[:5])
    # exit()
    train_matrix = np.array(train_df)
    test_matrix = np.array(test_df)
   
    for x in test_df:
        
        cur_dist = gen_eucl_dist(x, train_matrix)
        
        sml_dist_ind = np.argpartition(cur_dist, k)[:k]
        closest_nei = np.take(solutions, sml_dist_ind)
        
        sum = 0
        for ind in closest_nei:
            if ind == 'yes':
                sum += 1
                
        average = sum / len(sml_dist_ind)
        
        if average >= 0.5:
            nn_classific_cls.append('yes')
        else:
            nn_classific_cls.append('no')
    
    return nn_classific_cls

def classify_nb(training_filename, testing_filename, overide_data=False):
    # P(E|Yes) = P(E_1|yes) + ... + P(E_n|yes)
    # P(yes|E) = ( P(E|yes)P(yes) ) / P(E)
    # (1/var sqrt(2 pi) ) * e ^ ( - (x - mean)^2 / 2 * (var^2))
    
    col_names = ['num_preg','gluc_conc','bld_prs','skn_thck','insulin','bmi','dia_ped_f','age','class']
    nn_classific_cls = []
    
    if overide_data == False:
        train_df, solutions = get_file_data(training_filename, False)
        test_df, s = get_file_data(testing_filename, True)
    else:
        train_df = training_filename[0]
        solutions = training_filename[1]
        test_df = testing_filename
    
    train_matrix = np.array(train_df)
    test_matrix = np.array(test_df)
    
    solutions = np.array(solutions)
    sol_mtx = np.where(solutions == 'yes', 1, 0)
    mean_std_arr = gen_mean_std_per_x(train_matrix, solutions)
        

    classifications = []
    P_yes = sum(sol_mtx)/len(sol_mtx)
    P_no = 1-P_yes
    
    for test in test_matrix:
        P_x_yes = prob_dens_func(test, mean_std_arr, 0)
        P_x_no = prob_dens_func(test, mean_std_arr, 1)
        P_yes_x = P_x_yes * P_yes
        P_no_x = P_x_no * P_no
            
        if P_yes_x >= P_no_x:
            classifications.append('yes')
        else:
            classifications.append('no')
        
    return classifications

def prob_dens_func(x, mean_std, yes_no):
    tot_P = 1
    for i, x_var in enumerate(x):
        # Calculate probability density
        mean = mean_std[i][yes_no][0]
        std = mean_std[i][yes_no][1]
        expon = - ((x_var - mean) ** 2) / (2 * (std**2))
        result = (1 / ( std * math.sqrt(2 * math.pi))) * (math.e ** expon)
        tot_P = tot_P * result
    return tot_P

def gen_mean_std_per_x(train_matrix, solutions):
    sol_mask_yes = np.where(solutions == 'yes', 0, 1) # 0 for keep, 1 for remove (for some reason??) (for yes's)
    sol_mask_no = np.where(solutions == 'yes', 1, 0) # 1 for keep, 0 for remove (for no's)
    
    mean_std_arr = []
    
    for i in range(len(train_matrix[0])):
        
        rand_var = train_matrix[:,i]
        x_yes = ma.masked_array(rand_var, mask=sol_mask_yes)
        sub_yes = x_yes[~x_yes.mask]
        x_no = ma.masked_array(rand_var, mask=sol_mask_no)
        sub_no = x_no[~x_no.mask]
        
        mean_yes = np.mean(sub_yes)
        mean_no = np.mean(sub_no)
        std_yes = np.std(sub_yes)
        std_no = np.std(sub_no)
        new_entry = [(mean_yes, std_yes), (mean_no, std_no)]
        mean_std_arr.append(new_entry)
    
    return mean_std_arr

def get_resutls(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

def main():
    
    classif = classify_nn("train.data", "test.data", 5)
    classif = classify_nb("train.data", "test.data")
    print(len(classif))
    test_results = get_resutls("test.results")
    
    tr_pos = 0
    fa_pos = 0
    tr_neg = 0
    fa_neg = 0
    
    for i in range(len(classif)):
        if classif[i] == 'yes' and test_results[i] == 'yes':
            tr_pos += 1
        elif classif[i] == 'yes' and test_results[i] == 'no':
            fa_pos += 1
        elif classif[i] == 'no' and test_results[i] == 'yes':
            fa_neg += 1
        elif classif[i] == 'no' and test_results[i] == 'no':
            tr_neg += 1
    
    print(f'{"":<8}{"Real":<8}')
    print(f'{"Guess":<8}{"True":<8}{"False":<8}')
    print(f'{"True":<8}{tr_pos:<8.2f}{fa_pos:<8.2f}')
    print(f'{"False":<8}{fa_neg:<8.2f}{tr_neg:<8.2f}')
    
if __name__ == "__main__":
    # Main for testing purposes
    main()
    pass