#coding:utf-8
"""
* @auther tzw
* @date: 20180129
"""
import numpy as np
import argparse, os, csv
import pandas as pd
from sklearn.cross_validation import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from functools import reduce
from sklearn.grid_search import GridSearchCV

def main():
    parser = argparse.ArgumentParser(description='Random Forest for parkinsons')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--input_file', '-i', default='data/parkinsons.csv',
                        help='Path to input csv file')
    parser.add_argument('--out', '-o', default='results',
                        help='Directory to output the result')
    args = parser.parse_args()

    feature_list =['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',	'MDVP:Jitter(Abs)',
                   'MDVP:RAP'   , 'MDVP:PPQ'    , 'Jitter:DDP'  , 'MDVP:Shimmer'  ,	'MDVP:Shimmer(dB)',
                   'Shimmer:APQ3','Shimmer:APQ5', 'MDVP:APQ'    , 'Shimmer:DDA'   , 'NHR',
                   'HNR',	      'RPDE',	      'DFA',	      'spread1',	    'spread2',
                   'D2',	      'PPE']

    parameters_list = {
        'n_estimators' : [50,100,200,300,400,500],# #tree
        'max_depth'    : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],# depth of tree
        #"max_features" : [1, 3, 5, 10],#分割候補数
        'random_state' : [0, 10, 100],# seed
        #'n_jobs'       : [-1],#並列処理するCPUの数
    }

    # Read Features
    input_file_path = os.path.join(args.base, args.input_file)
    data_df = pd.read_csv(input_file_path, header=0)
    targets = data_df['status'].values
    features = data_df[feature_list].values

    loo = LeaveOneOut(len(features))
    results = []

    #LOO loop
    for train_indexes, test_indexes in loo:
        print('Test No. {}'.format(test_indexes[0]))
        # separate data
        train_features = features[train_indexes]
        train_targets = targets[train_indexes]
        test_features = features[test_indexes]
        test_targets = targets[test_indexes]

        # Grid Search
        model = RandomForestClassifier(n_jobs=-1)
        forest_grid = GridSearchCV(model, param_grid=parameters_list, cv=2, n_jobs=-1)
        forest_grid.fit(train_features, train_targets)
        forest_grid_best = forest_grid.best_estimator_
        #print('Best Model Parameter: {}'.format(forest_grid.best_params_))
        #print(forest_grid_best)

        # Training
        forest_grid_best = forest_grid_best.fit(train_features, train_targets)

        # Predict
        output = forest_grid_best.predict(test_features)
        print('Predict Result: {}'.format(output[0]))
        print('Ground Truth: {}'.format(test_targets[0]))

        # Judgment
        results.append(test_targets==output)

        # Save model & log
        #save_dir = args.out + '/' + str(test_indexes[0])
        #if not os.path.exists(save_dir):
        #    os.makedirs(save_dir)
        #joblib.dump(model, save_dir+'/clf.pkl')
        result_dir = os.path.join(args.base, args.out)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        with open('{}/result.csv'.format(result_dir),'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([output[0], test_targets[0]])


    # Result
    N = len(features)
    correct = reduce(lambda n, o: n + 1 if o else n, results, 0)
    msg = 'Correct: {0}/{1}'.format(correct, N)
    print(msg)
    failed = reduce(lambda n, o: n + 1 if not o else n, results, 0)
    msg = 'Failed: {0}/{1}'.format(failed, N)
    print(msg)
    correct_rate = (float(correct) / N) * 100
    msg = 'Correct Rate: {0}%'.format(correct_rate)
    print(msg)


if __name__ == '__main__':
    main()
