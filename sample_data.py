from cgi import test
import numpy as np
import numpy.ma as ma

def randomize_data(filename, ratio=0.7):
    
    np.random.seed(2)
    train_arr = []
    test_arr = []
    test_solutions = []
    with open(filename, 'r') as f:
        headers = f.readline()
        data = f.readlines()
        n_data = len(data)
        rand_indx = np.random.choice([True, False], n_data, p=[ratio, 1-ratio])
        
        for i, line in enumerate(data):
            cur_row = line.split(',')
            if len(line) < 4:
                continue
            if rand_indx[i] == True:
                train_arr.append(cur_row)
            else:
                # print(cur_row[-1])
                test_solutions.append(cur_row[-1])
                new_col = cur_row[-2] + '\n'
                cur_row = cur_row[:-2]
                cur_row.append(new_col)
                test_arr.append(cur_row)
        # Remove random white space at end of test arr
        if train_arr[-1] == ['\n']:
            train_arr = train_arr[:-2]
            
        if test_arr[-1] == ['\n']:
            test_arr = test_arr[:-1]
        
    with open("train.data", 'w') as f:
        for row in train_arr:
            f.write(','.join(row))
    
    with open("test.data", 'w') as f:
        for row in test_arr:
            f.write(','.join(row))
    
    with open("test.results", 'w') as f:
        f.writelines(test_solutions)
            
            
        

if __name__ == "__main__":
    randomize_data('normalize_data.data')