import numpy as np
from funcs import *
import sys

''' Bandit-Based Task Assignment '''
def BBTA(data, n_explore=0, query='margin', beta=0, maxlevel_budget=15):
    n_task = data['n_task'] # number of tasks
    n_worker = data['n_worker'] # number of workers
    tasks = data['tasks'] # labeling tasks
    max_budget = n_task * maxlevel_budget # maximum budget
    n_rec = maxlevel_budget - 1 # number of accuracy records
    ctxs = np.unique(tasks) # contexts of tasks
    n_ctx = len(ctxs) # number of contexts
    den_ctx, den_task = calc_density_ctx(tasks, ctxs) # densities of contexts
    loss_cu = np.zeros((n_ctx, n_worker)) # cumulative losses
    ctx_counts = np.zeros(n_ctx) # appearance counts for all contexts 
    mat_Y = np.zeros((n_task, n_worker))  # label matrix
    scores = np.power(den_task, beta) # sampling scores
    y_hat = np.zeros(n_task) # aggregated labels
    y_hats = np.zeros((n_task, n_rec)) # record of aggregated labels
    available_tasks = np.arange(n_task) # indices of available tasks
    budget_explore = 0 # consumed budget in exploration phase
    t_each_task = max_budget # the step where each task has at least one label
    flag_each_task = False # the flag

    ''' pure exploration phase, select n_explore tasks per ctx '''
    if n_explore > 0:
        for i_ctx in xrange(n_ctx):
            ctx = ctxs[i_ctx]

            ''' randomly select n_explore tasks for each ctx '''
            indices_t, = (tasks == ctx).nonzero()
            np.random.shuffle(indices_t)
            for i in xrange(n_explore):
                i_task = indices_t[i]
                mat_Y[i_task] = get_labels(data['Y'], # simulate labeling
                                      i_task, range(n_worker))

                ''' aggregate labels from all workers '''
                scores[i_task], y_hat[i_task] = label_aggregation(mat_Y[i_task], 
                    np.ones(n_worker), query, den_ctx[i_ctx], beta)
                loss_cu[i_ctx] = loss_cu[i_ctx] + \
                                    (y_hat[i_task] != mat_Y[i_task])
                available_tasks = np.delete(available_tasks, 
                                  (available_tasks == i_task).nonzero()[0][0])
                budget_explore += n_worker

    ''' task assignment phase '''
    for t in xrange(budget_explore + 1, max_budget + 1):
        if not available_tasks.size:
            print 'Stopped due to labeling completion!'
            break
        
        ''' pick the task according to the query strategy '''
        if query == 'random':
            idx = np.random.randint(len(available_tasks))
        else:
            idx = np.argmax(scores[available_tasks])
        i_task = available_tasks[idx]
        ctx = tasks[i_task]
        
        ''' calculate weights '''
        i_ctx = (ctxs == ctx).nonzero()[0][0]
        ctx_counts[i_ctx] += 1
        eta = np.sqrt(np.log(n_worker) / ctx_counts[i_ctx] / n_worker)
        w = np.exp(-eta * loss_cu[i_ctx])
        
        ''' select a worker '''
        indices_w, = (mat_Y[i_task] == 0).nonzero()
        p = w[indices_w] / np.sum(w[indices_w])
        ip = np.random.multinomial(1, p).nonzero()[0][0]
        i_worker = indices_w[ip]

        mat_Y[i_task, i_worker] = get_labels(data['Y'], # simulate labeling
                                      i_task, i_worker)
        
        ''' aggregate labels '''
        scores[i_task], y_hat[i_task] = label_aggregation(mat_Y[i_task], 
            w, query, den_ctx[i_ctx], beta)
        
        loss = int(y_hat[i_task] != mat_Y[i_task, i_worker]) # feedback
        
        ''' update cumulative loss and confidence scores'''
        if np.sum(mat_Y[i_task] == 0) == 0:
            available_tasks = np.delete(available_tasks, 
                              (available_tasks == i_task).nonzero()[0][0])
        if loss:
            loss_cu[i_ctx, i_worker] += (loss / p[ip])
            w = np.exp(-eta * loss_cu[i_ctx])
            for i in (tasks == ctx).nonzero()[0]:
                scores[i], y_hat[i] = label_aggregation(mat_Y[i], 
                    w, query, den_ctx[i_ctx], beta)

        ''' record the accuracy '''
        if (t % n_task == 0) and (t >= 2 * n_task):
            y_hats[:, t/n_task - 2] = y_hat

        ''' check if each task has at least one label '''
        if not flag_each_task:
            flag_each_task = is_each_task(mat_Y)
            t_each_task = t 

    return y_hats, t_each_task