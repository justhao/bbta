import numpy as np 

def calc_density_ctx(tasks, ctxs):
	m1, m2 = np.meshgrid(ctxs, tasks)
	den_ctx = 1. * np.sum(m1 == m2, axis=0) / len(tasks)
	den_task = np.array([den_ctx[(ctxs == c).nonzero()[0][0]] for c in tasks])
	return den_ctx, den_task

def get_labels(Y, i_task, i_worker):
    return Y[i_task, i_worker]

def label_aggregation(labels, weights, query, density, beta):
    ind_pos, = (labels == 1).nonzero()
    ind_neg, = (labels == -1).nonzero()
    ind_0, = (labels == 0).nonzero()
    sum_weights = np.sum(weights)
    sum_weights_pos = np.sum(weights[ind_pos])
    sum_weights_neg = np.sum(weights[ind_neg])
    sum_weights_0 = np.sum(weights[ind_0])
    score_pos = sum_weights_pos / sum_weights
    score_neg = sum_weights_neg / sum_weights
    score_0 = sum_weights_0 / sum_weights

    if query == 'margin':
        conf_score = abs(score_pos - score_neg)
    elif query == 'lc':
        conf_score = max(score_pos, score_neg)
    else:
    	conf_score = 0
    
    score = (1-conf_score) * np.power(density, beta)

    if score_pos == score_neg:
        y_hat = (np.random.randint(2) * 2 - 1)
    elif score_pos > score_neg:
        y_hat = 1
    else:
        y_hat = -1

    return score, y_hat

def is_each_task(mat_Y):
	n_task, n_worker = mat_Y.shape
	return (np.sum(np.sum(mat_Y == 0, axis=1) == n_worker) == 0)

