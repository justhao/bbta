import numpy as np
from funcs import *
import sys

''' A Naive Baseline of Task Assignment '''
def random_assignment(data, maxlevel_budget=15):
    n_task = data['n_task'] # number of tasks
    n_worker = data['n_worker'] # number of workers

    max_budget = n_task * maxlevel_budget # maximum budget
    n_rec = maxlevel_budget - 1 # number of accuracy records
    mat_Y = np.zeros((n_task, n_worker))  # label matrix
    y_hat = np.zeros(n_task) # aggregated labels
    y_hats = np.zeros((n_task, n_rec)) # record of aggregated labels
    available_tasks = np.arange(n_task) # indices of available tasks

    t_each_task = max_budget # the step where each task has at least one label
    flag_each_task = False # the flag

    ''' task assignment '''
    for t in xrange(1, max_budget + 1):
        if not available_tasks.size:
            print 'Stopped due to labeling completion!'
            break
        
        ''' randomly pick a task '''
        idx = np.random.randint(len(available_tasks))
        i_task = available_tasks[idx]
        
        ''' randomly select a worker '''
        indices_w, = (mat_Y[i_task] == 0).nonzero()
        idx = np.random.randint(len(indices_w))
        i_worker = indices_w[idx]

        mat_Y[i_task, i_worker] = get_labels(data['Y'], # simulate labeling
                                      i_task, i_worker)
        
        ''' aggregate labels '''
        _, y_hat[i_task] = label_aggregation(mat_Y[i_task],
            np.ones(n_worker), 'random', 1, 0)
        
        ''' check whether fully labeled '''
        if np.sum(mat_Y[i_task] == 0) == 0:
            available_tasks = np.delete(available_tasks, 
                              (available_tasks == i_task).nonzero()[0][0])

        ''' record the accuracy '''
        if (t % n_task == 0) and (t >= 2 * n_task):
            y_hats[:, t/n_task - 2] = y_hat

        ''' check if each task has at least one label '''
        if not flag_each_task:
            flag_each_task = is_each_task(mat_Y)
            t_each_task = t 

    return y_hats, t_each_task