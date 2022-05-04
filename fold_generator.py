import sys
import random

def main():
    
    filename = sys.argv[1]
    
    yes_arr = []
    no_arr = []
    
    # Open the data file and read everything in
    with open(filename, 'r') as f:
        # headers = f.readline()
        
        # Read each line and do accordingly sort yes or no
        while True:
            line = f.readline()
            if line == '\n' or line == '':
                break
            line = line.strip()
            row = line.split(',')
            if row[-1] == 'yes':
                yes_arr.append(row)
            else:
                no_arr.append(row)
    
    # Shuffle the Arrays
    random.shuffle(yes_arr)
    random.shuffle(no_arr)
    
    folds = [[],[],[],[],[],[],[],[],[],[]]
    
    count = 0
    for row in yes_arr:
        folds[count].append(row)
        # Do counter stuff
        count += 1
        if count == 10:
            count = 0
    
    count = 0
    for row in no_arr:
        folds[count].append(row)
        # Do counter stuff
        count += 1
        if count == 10:
            count = 0
    
    
    # Print folds to file
    with open('data-folds.csv', 'w') as f:
        
        for fold_ind in range(len(folds)):
            f.write(f'fold{fold_ind+1}\n')
            for row in range(len(folds[fold_ind])):
                line = ','.join(folds[fold_ind][row])
                line += '\n'
                f.write(line)
            if fold_ind != 9:
                f.write('\n')

main()