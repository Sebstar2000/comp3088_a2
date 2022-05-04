import sys
from program1 import classify_nn
from program1 import classify_nb

def convert_fold(fold):
    pass

def read_folds(filename):
    all_folds = []
    
    with open(filename, 'r') as f:
        
        for i in range(10):
            
            header = f.readline()
            cur_fold = []
            while True:
                line = f.readline()
                if line == '' or line == '\n':
                    break
                
                line = line.strip()
                line = line.split(',')
                line_new = [float(i) for i in line[:-1]]
                line_new.append(line[-1])
                cur_fold.append(line_new)
                
            all_folds.append(cur_fold)
            
    return all_folds

def convert_fold_train_test(data, test_indx):
    train = []
    train_sol = []
    test = []
    test_sol = []
    
    for i, fold in enumerate(data):
        if i == test_indx:
            for row in fold:
                test_sol.append(row[-1])
                test.append(row[:-1])
        else:
            for row in fold:
                train_sol.append(row[-1])
                train.append(row[:-1])
    return train, train_sol, test, test_sol

def main():
    
    filename = sys.argv[1]
    folds = read_folds(filename)
    
    cor_class_arr = []
    
    for i in range(10):
        train, tra_sol, test, test_sol = convert_fold_train_test(folds, i)
        
        nn_sol = classify_nb((train, tra_sol), test, True)
        
        count = 0
        for sol in range(len(nn_sol)):
            if nn_sol[sol] == test_sol[sol]:
                count = count + 1
        
        
        cor_class_arr.append(count/len(nn_sol))
    
    print((sum(cor_class_arr)/len(cor_class_arr))*100)
    
     

if __name__ == '__main__':
    main()