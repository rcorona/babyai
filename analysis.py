import pickle
import os
import pdb
import numpy as np

# Run analysis for each folder. 
for v in os.listdir('results'):

    print('{}\n-----------------------'.format(v))

    # Load task results.  
    for split in ('atomic', 'compositional'):
        try: 
            results = pickle.load(open(os.path.join('results', v, '{}.pkl'.format(split)), 'rb'))
            print('{}\n'.format(split.upper()))

            for task in results: 
                print('Task: {} SR: {} STD: {}\n'.format(task, results[task]['success_per_episode']['mean'], results[task]['success_per_episode']['std']))

            avg_success = np.mean([results[task]['success_per_episode']['mean'] for task in results])

            print('Avg. SR: {}\n'.format(avg_success))
        except: 
            pass
