import torch
import gc

def modified_Fisher_inverse(model, 
                 output :torch.Tensor, 
                 y:torch.Tensor,
                 output_true :torch.Tensor, 
                 y_true:torch.Tensor,
                 modify = True):
    """calulcates each layerwise component and returns a torch.Tensor representing the NTK
    
        parameters:
            model: a torch.nn.Module object. Must terminate to a single neuron output
            output: the final single neuron output of the model evaluated on some data
            y: the labels
            output: the final single neuron output of the model evaluated on true date (Here we use validation set)
            y: the labels of true data (Here we use validation set)

        returns:
            NTK: a torch.Tensor representing the emprirical neural tangent kernel of the model
    """
    threshold = 1e-8
    threshold2 = 1e4
    sigma2 = 1e-2
    device = y.device
    NTK = False
    
    # calculate the empirical Jacobian prepared for Fisher on training set 
    if len(output.shape) > 1:
        raise ValueError('y must be 1-D, but its shape is: {}'.format(output.shape))
    
    params_that_need_grad = []
    for param in model.parameters():
        if param.requires_grad:
            params_that_need_grad.append(param.requires_grad)
            #first set all gradients to not calculate, time saver
            param.requires_grad = False
        else:
            params_that_need_grad.append(param.requires_grad)
    J_list = []
    for i,z in enumerate(model.named_parameters()):
        if not(params_that_need_grad[i]): #if it didnt need a grad, we can skip it.
            continue
        name, param = z
        param.requires_grad = True #we only care about this tensors gradients in the loop
        this_grad=[]
        for i in range(len(output)): #first dimension must be the batch dimension
            model.zero_grad()
            output[i].backward(create_graph=True)
            this_grad.append(param.grad.detach().reshape(-1).clone())

        J_layer = torch.stack(this_grad).detach_() 
        J_list.append(J_layer)  # [N x P matrix] #this will go against our notation, but I'm not adding
        # if (type(NTK) is bool) and not(NTK):
        #     NTK = J_layer @ J_layer.T # An extra transpose operation to my code for us to feel better
        # else:
        #     NTK += J_layer @ J_layer.T

        param.requires_grad = False
    J = torch.cat(J_list, dim=1).to(device)
    # J = copy.deepcopy(J.detach_().clone().cpu())
    # J = J.to(device)
    # J = copy.deepcopy(J)
    sample_num = J.shape[0]
    param_num = J.shape[1]
    #reset the model object to be how we started this function
    for i,param in enumerate(model.parameters()):
        if params_that_need_grad[i]:
            param.requires_grad = True #

    # calculate the svd decomposition of J
    with torch.no_grad():
        U ,S, Vh = torch.linalg.svd(J)
    # print('min of S', torch.min(S))
    # U = U.cpu()
    V = (Vh.T)
    # J = J.cpu()
    
    # S = S.cpu()
    # print('max of V', torch.max(V))
    del Vh, J
    gc.collect()
    torch.cuda.empty_cache()

    if modify == True:
        # calculate the gradient in function space on training set 
        alpha = (output - y)
        # calculate the diagonal of empirical Fisher's eigenvalues
        Q_train = torch.pow(S, 2)/sample_num#/torch.tensor(sigma2, device=device) shape [sample_num], cut for the following computation since the rest elements will be mutiplied by zero
        # calculate the empirical gradient in function space projected onto eigenspace of NTK 
        aTu = (alpha.T @ U)  

        del alpha, U
        gc.collect()
        torch.cuda.empty_cache()
        # calculate the empirical gradient in parameter space 
        G_train = aTu * S/sample_num #shape [sample_num], cut for the following computation since the rest elements will be mutiplied by zero

        # Do same thing on the validation set representing for true data 
        if len(output_true.shape) > 1:
            raise ValueError('y must be 1-D, but its shape is: {}'.format(output_true.shape))
        
        J_true = []
        #how do we parallelize this operation across multiple gpus or something? that be sweet.
        for i,z in enumerate(model.named_parameters()):
            if not(params_that_need_grad[i]): #if it didnt need a grad, we can skip it.
                continue
            name, param = z
            param.requires_grad = True #we only care about this tensors gradients in the loop
            this_grad=[]
            for i in range(len(output_true)): #first dimension must be the batch dimension
                model.zero_grad()
                output_true[i].backward(create_graph=True)
                this_grad.append(param.grad.detach().reshape(-1).clone())

            J_true_layer = torch.stack(this_grad) # [N x P matrix] #this will go against our notation, but I'm not adding
            J_true.append(J_true_layer)
            # if (type(NTK) is bool) and not(NTK):
            #     NTK = J_layer @ J_layer.T # An extra transpose operation to my code for us to feel better
            # else:
            #     NTK += J_layer @ J_layer.T

            param.requires_grad = False
        J_true = torch.cat(J_true, dim=1).to(device).detach()
        sample_num_t = J_true.shape[0]
        param_num_t  = J_true.shape[1]
        #reset the model object to be how we started this function
        for i,param in enumerate(model.parameters()):
            if params_that_need_grad[i]:
                param.requires_grad = True #
        
        
        # calculate the gradient in function space on training set 
        alpha_true = output_true - y_true
        # calculate the empirical Fisher multiplied by VT * V
        Q_true = torch.diag(V.T @ J_true.T @ J_true @ V)[:sample_num]/sample_num_t#/sigma2   # sigma2 is the variance of Gaussian assumption #shape [sample_num], cut for the following computation since the rest elements will be mutiplied by zero
        # calculate the empirical gradient in parameter space 
        G_true = (alpha_true.T @ J_true @ V)[:sample_num]/sample_num_t #shape [sample_num], cut for the following computation since the rest elements will be mutiplied by zero
        del J_true
        gc.collect()
        torch.cuda.empty_cache()
        # calculate the modification criterion 
        '''
        q_i \frac{1}{\lambda_i^2} - \frac{2l_i}{\alpha(\mathcal{X},\mathcal{Y})^\top u_i}\frac{1}{\lambda_i} -\frac{1}{N} > 0
        '''
        Q = (Q_true- Q_train).to(device)
        L = (G_true - G_train).to(device)
        del Q_train, Q_true, G_train, G_true, alpha_true
        gc.collect()
        torch.cuda.empty_cache()
        # S = S
        # print('Q shape', Q.device, '\n')
        # print('aTu shape', aTu.device, '\n')
        # print('L shape', L.device, '\n')
        # print('S[:sample_num] shape', S.device, '\n')
        Q = Q.to(device)
        aTu = aTu.to(device)
        L = L.to(device)
        S = S.to(device)
        V = V.to(device)
        # print('model device', model)
        # print('min of S', torch.min(S))
        tmp = torch.pow(1/S,2)
        S = torch.where(tmp>threshold**2, S, 0)
        criterion = Q*torch.pow(aTu, 2) - 2* L* aTu * S - torch.pow(aTu, 2) * torch.pow(S, 2) /sample_num
        S = torch.where((criterion > 0)|(tmp<=threshold**2), 0, tmp)*sample_num*torch.tensor(sigma2,device=device)
        # S = torch.where((tmp<=threshold**2), 0, tmp)*sample_num*torch.tensor(sigma2,device=device)
        diag_of_modified_Fisher_inverse =  torch.cat([S, torch.zeros(param_num-S.shape[0], device=device)])  # sigma2 is the variance of Gaussian assumption
        diag_of_modified_Fisher_inverse = torch.where(diag_of_modified_Fisher_inverse>threshold2, threshold2, diag_of_modified_Fisher_inverse)
        F_inverse_modified = (V) @ (diag_of_modified_Fisher_inverse *  V.T)
        # print('max of F', torch.max(F_inverse_modified))
        # print('mean of F', torch.mean(F_inverse_modified))
        del V, Q ,aTu, L, S, tmp
        gc.collect()
        torch.cuda.empty_cache()
        return F_inverse_modified
        
    if modify==False:
        # calculate the empirical Fisher multiplied with sigma_0^2
        tmp = torch.pow(1/S,2)
        S = torch.where(tmp>threshold**2, tmp, 0)
        S = torch.cat([S, torch.zeros(param_num - S.shape[0], device=device)])
        # print('max S', torch.max(S))
        S = torch.where(S>threshold2, threshold2, S)
        F = (V) @ (S*V.T) *sample_num*torch.tensor(sigma2,device=device)  # sigma2 is the variance of Gaussian assumption
        # print('mean of F', torch.mean(F))
        del S, U, tmp
        gc.collect()
        torch.cuda.empty_cache()
        
        return F
