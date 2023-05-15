import torch
import numpy as np    
gamma_train = torch.rand(256).numpy()
gamma_valid = torch.rand(64).numpy() 
gamma_test  = torch.rand(64).numpy() 
gamma_monte = torch.rand(1024).numpy()

np.save('./data/without_perturb/gamma_train.npy', gamma_train)
np.save('./data/without_perturb/gamma_test.npy', gamma_test)
np.save('./data/without_perturb/gamma_valid.npy', gamma_valid)
np.save('./data/without_perturb/gamma_monte.npy', gamma_monte)

sigma = 'infty'
if sigma != 'infty':
    gamma_train = gamma_train*np.exp(-np.power(1-gamma_train,2)/sigma)
np.save(f'./data/perturb/{sigma}/gamma_train.npy', gamma_train)
np.save(f'./data/perturb/{sigma}/gamma_test.npy', gamma_test)
np.save(f'./data/perturb/{sigma}/gamma_valid.npy', gamma_valid)
np.save(f'./data/perturb/{sigma}/gamma_monte.npy', gamma_monte)

