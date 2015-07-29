import sys
from time import strftime
import numpy as np
from scipy.stats import sem 
import matplotlib.pyplot as plt
import seaborn as sns
from BBTA import BBTA
from naive_baseline import random_assignment

np.random.seed(sum(map(ord, 'BBTA')))

maxlevel_budget = 15
n_rec = maxlevel_budget - 1
n_run = 10
n_method = 3

''' make toy data '''
n_task = 200 # task number
n_worker = 20 # worker number
n_ctx = 2 # context number

# tasks with context
tasks = np.random.randint(n_ctx, size=n_task) 

# true labels of tasks, -1 or 1
z = np.random.randint(2, size=n_task) * 2 - 1 
Z = np.tile(np.c_[z], [1, n_rec])
ZY = np.tile(np.c_[z], [1, n_worker])

ctx_ac = [0.9, 0.6] # context-dependent accuracy

# familiar context for each worker
familiar = np.random.randint(n_ctx, size=n_worker) 

# simulate worker labels
m_familiar, m_tasks = np.meshgrid(familiar, tasks)
Y = (m_familiar == m_tasks)
Y = Y * ctx_ac[0] + (-Y) * ctx_ac[1]
Y = (np.random.rand(n_task, n_worker) < Y)
Y = Y * ZY + (-Y) * (-ZY)

# wrap
toydata = {}
toydata['n_task'] = n_task
toydata['n_worker'] = n_worker
toydata['tasks'] = tasks
toydata['Y'] = Y

# plot params
pal = sns.color_palette()
sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1.7, rc={'figure.figsize': (6, 5),
												'lines.linewidth': 2.5})

params = [('-', pal[0], 0, 'BBTA, N\'=0'),
		  ('-', pal[2], 1, 'BBTA, N\'=1'),
		  ('--', 'k',   0, 'Naive Baseline')]

''' task assignment '''
plt.figure()

for i_method in xrange(n_method):
	acs = []
	linestyle, color, n_explore, label = params[i_method]
	for i_run in xrange(n_run):
		if i_method != n_method - 1:
			y_hats, t_each_task = BBTA(toydata, n_explore=n_explore)
		else:
			y_hats, t_each_task = random_assignment(toydata)
		ac = sum(y_hats == Z) / float(n_task)
		acs.append(ac)
		print 'Run {0:2}, '.format(i_run+1) + strftime('%H:%M:%S') \
			  + ', ' + params[i_method][-1]
	mean = np.mean(acs, axis=0)
	std = sem(acs, axis=0)
	x_axis = np.linspace(2*n_task, maxlevel_budget*n_task, n_rec)
	ind = (x_axis >= t_each_task)
	plt.errorbar(x_axis[ind], mean[ind], yerr=std[ind], 
		ls=linestyle, color=color, label=label)
	print params[i_method][-1] + ' finished.'
plt.xlabel('Budget')
plt.ylabel('Accuracy')
plt.title('Toy Data')
plt.legend(loc='lower right')
plt.show()