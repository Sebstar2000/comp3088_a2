import sys
from program1 import classify_nn
from program1 import classify_nb

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

def convert_fold_train_test(data, test_indx, CFS=None):
    train = []
    train_sol = []
    test = []
    test_sol = []
    
    for i, fold in enumerate(data):
        
        if i == test_indx:
            for row in fold:
                # Feature selections stuff
                cur_row = row
                if CFS != None:
                    cur_row = [row[x] for x in CFS]
                test_sol.append(cur_row[-1])
                test.append(cur_row[:-1])
        else:
            for row in fold:
                # Feature selections stuff
                cur_row = row
                if CFS != None:
                    cur_row = [row[x] for x in CFS]
                train_sol.append(cur_row[-1])
                train.append(cur_row[:-1])
    return train, train_sol, test, test_sol

def main(CFS=None):
    
    filename = sys.argv[1]
    folds = read_folds(filename)
    
    percent_class_arr = []
    nn1_fold_accuracy = []
    nn5_fold_accuracy = []
    nb_fold_accuracy = []
    
    for i in range(10):
        if CFS != None:
            train, tra_sol, test, test_sol = convert_fold_train_test(folds, i, CFS)
        else:
            train, tra_sol, test, test_sol = convert_fold_train_test(folds, i)
        
        # NN1 Test and accuracy check
        nn1_sol = classify_nn((train, tra_sol), test, 1, True)
        
        count = 0
        for sol in range(len(nn1_sol)):
            if nn1_sol[sol] == test_sol[sol]:
                count = count + 1
        
        nn1_fold_accuracy.append(count/len(nn1_sol))
        
        # NN5 Test and accuracy check
        nn5_sol = classify_nn((train, tra_sol), test, 5, True)
        
        count = 0
        for sol in range(len(nn5_sol)):
            if nn5_sol[sol] == test_sol[sol]:
                count = count + 1
        
        nn5_fold_accuracy.append(count/len(nn5_sol))
        
        # NB Test and accuracy check
        nb_sol = classify_nb((train, tra_sol), test, True)
        
        count = 0
        for sol in range(len(nb_sol)):
            if nb_sol[sol] == test_sol[sol]:
                count = count + 1
        
        nb_fold_accuracy.append(count/len(nb_sol))
        
    if CFS == None:
        print('Solutions without feature selection')
    else:
        print('Solutions with feature selection: 2,5,6,7,8 (from weka)')
        
    print('Nearest Neighbour 1 avg. accu',(sum(nn1_fold_accuracy)/len(nn1_fold_accuracy))*100)
    print('Nearest Neighbour 5 avg. accu',(sum(nn5_fold_accuracy)/len(nn5_fold_accuracy))*100)
    print('Naive Bayes avg. accu', (sum(nb_fold_accuracy)/len(nb_fold_accuracy))*100)
    
     

if __name__ == '__main__':
    
    # CFS Selection from weka: 2,5,6,7,8
    CFS = [1, 4, 5, 6, 7, 8] # The above CFS (-1) plus 8 for class
    
    # Normal
    # main()
    # CFS version
    main(CFS)