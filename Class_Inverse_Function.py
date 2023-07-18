import torch
import numpy as np
nn = torch.nn
import matplotlib.pyplot as plt

class Inverse_fct:
    """
    Gives the inverse of a given function v
    """
    def __init__(self,m,xtrue=None,vx=None):
        
        if xtrue==None:
             raise Exception("Must provide antecedent of your function")
        if vx==None:
             raise Exception("Must provide values of your function")
        self.fct_antecedent=xtrue
        self.fct_value=vx
        self.model = m
        self.totalEpochs = 0

   
    def loss(self, xpred, xreal):
        loss_value = (xpred - xreal) ** 2
        return loss_value
    
    def fit(self, getOptim,batch_size=500,epochs=50, vx=None, xtrue=None, lr=1e-3,evolution=False,dynamic=False,lr_adapt=False):
        if vx is None:
            vx = self.fct_value
        if xtrue is None:
            xtrue = self.fct_antecedent
        
        
        optimizer = getOptim(self.model,lr)
        print("Parameters before the run : ") 
        
        if lr_adapt:
            scheduler = schedulerMap[ 'CyclicLR_triang' ](optimizer,lr)
            #scheduler = schedulerMap[ 'CyclicLR_triang2' ](optimizer,lr)
            #scheduler = schedulerMap[ 'CyclicLR_exp' ](optimizer,lr,gamma=0.8)
            #scheduler = schedulerMap[ 'StepLR' ](optimizer)
            #scheduler = schedulerMap[ 'Cos_restart' ](optimizer)
            #scheduler = schedulerMap[ 'Exp_LR' ](optimizer)
            self.lr=scheduler.get_last_lr()
        
       
        # Initialize lists to store loss and learning rate values
        gradtot=[]
        losses = []
        learning_rates=[]
        if dynamic:
            fig, axs = plt.subplots(3)
            fig.suptitle(f"Number of parameters {sum([t.numel() for t in self.model.parameters()])}. Scheduler :   {scheduler.__class__.__name__}. ")
            axs[0].set_ylabel("Loss")
            axs[1].set_ylabel("Learning Rate")
            axs[1].set_xlabel("Epoch")
        
        plt.ion()  # Turn on interactive mode for dynamic plotting
        
        nx = len(vx)
        if batch_size>nx:
            batch_size=int(nx/2)  
        for epoch in range(epochs):
            grad=[]
            epoch_losses = []
            epoch_learning_rates = []
            # setup shuffled list of indices at which to pick  x values during training :
            xi = np.arange(nx).reshape( (nx,1) ) 
            np.random.shuffle(xi) #on mélange son contenu
            for start in range(0,xi.shape[0],batch_size):
                stop = start + batch_size
                
                vx_batch_train = vx[torch.from_numpy(xi[start:stop])].reshape(-1,1)
                xtrue_batch_train = xtrue[torch.from_numpy(xi[start:stop])].reshape(-1,1)
                
                # calculate the loss
                loss = self.trainStep( vx_batch_train, xtrue_batch_train, optimizer)
                optimizer.zero_grad()
                # Update the model weights
                loss.backward()
                optimizer.step()
                
                if dynamic:
                    # Collect gradients of all parameters
                    all_gradients = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            all_gradients.append(param.grad.view(-1))
                    
                    # Calculate the norm of each gradient and take the average
                    gradient_norms = [torch.norm(gradient) for gradient in all_gradients]
                    grad.append( sum(gradient_norms) / len(gradient_norms) )
                    # Append loss and learning rate values for the current batch
                    epoch_losses.append(loss.item())
                    epoch_learning_rates.append(scheduler.get_last_lr()[0])
            
            if lr_adapt:
                # modify if necessary at every epoch
                scheduler.step()
                self.lr=scheduler.get_last_lr()
            if dynamic:
                # Append the average loss and learning rate values for the epoch
                losses.append(sum(epoch_losses) / len(epoch_losses))
                learning_rates.append(sum(epoch_learning_rates) / len(epoch_learning_rates))
                gradtot.append( (sum(grad) / len(grad)) )
                # Update the plot with the last 200 epochs
                start_epoch = max(0, epoch - 199)
                axs[0].cla()  # Clear the plot
                axs[0].plot(range(start_epoch, epoch + 1), losses[start_epoch:], 'b-')
                axs[1].cla()  # Clear the plot
                axs[1].plot(range(start_epoch, epoch + 1), learning_rates[start_epoch:], 'r-')
                axs[2].cla()  # Clear the plot
                axs[2].plot(range(start_epoch, epoch + 1), gradtot[start_epoch:], 'r-')
                plt.pause(0.01)
            print(f"Training epoch {epoch+self.totalEpochs}. Last losses :   {loss:.4e}")
            
        self.totalEpochs += epochs
        plt.ioff()
        plt.show()
        plt.close()
       
    def trainStep(self, vx,xtrue, optimizer):
        """Compute the derivative w.r.t inputs, then the diff equation & boundary conditions """
        
        vx = vx.clone().detach().requires_grad_(True)
        xtrue = xtrue.clone().detach()
        
        # Calculate the antecedent of the function        
        xpred = self.model(vx )
        
        optimizer.zero_grad()

        # Calculate the equation expression : 
        loss=  self.loss(xpred, xtrue).mean()

        return loss


    def plotModel2(self, otherF=None,XBC="x_bc",YBC="y_bc"):
        """
        To plot the right analytical solution for chosen bc
        """
        x=self.fct_value.detach().reshape(-1,1)
        u = self.model(x)
        plt.cla()
        plt.plot(x, u.detach(), label='f') 
        if otherF is not None:
            plt.plot(x, otherF(x).detach(), label='reference')
        plt.legend()
        plt.ioff()
        plt.show()
        plt.close()
        
        
def opti(name, l2=0,):
    """ short cut function to return an optimizer configured with given lr and l2 parameters """
    optiClass = optimizerMap[ name ]    
    opti = lambda model,scale_lr : optiClass(model.parameters(), lr= scale_lr, weight_decay=l2)
    return opti
    

# A convenient directory of known optimizers            
optimizerMap= dict(
    nada = torch.optim.NAdam,
    adam = torch.optim.Adam,
    radam = torch.optim.RAdam,
    SGD=torch.optim.SGD,
    #diffgrad = extoptim.DiffGrad,
    #lamb = extoptim.Lamb,
    #adamp = extoptim.AdamP
    )

# ======================= list of possible scheduler ==========================

schedulerMap=dict(
                StepLR=lambda optimizer : torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8),
                CyclicLR_triang = lambda optimizer,lr: torch.optim.lr_scheduler.CyclicLR(optimizer, 
                              base_lr = 0, # Initial learning rate which is the lower boundary in the cycle for each parameter group
                              max_lr = lr, # Upper learning rate boundaries in the cycle for each parameter group
                              step_size_up = 20,
                              step_size_down=150,# Number of training iterations in the increasing half of a cycle
                              mode = "triangular",),
                              #mode = "exp_range",
                              #gamma=0.8
                CyclicLR_triang2 = lambda optimizer,lr: torch.optim.lr_scheduler.CyclicLR(optimizer, 
                              base_lr = 0, # Initial learning rate which is the lower boundary in the cycle for each parameter group
                              max_lr = lr, # Upper learning rate boundaries in the cycle for each parameter group
                              step_size_up = 40,
                              step_size_down=1000,# Number of training iterations in the increasing half of a cycle
                              mode = "triangular2",),
                CyclicLR_exp = lambda optimizer,lr: torch.optim.lr_scheduler.CyclicLR(optimizer, 
                              base_lr = 0, # Initial learning rate which is the lower boundary in the cycle for each parameter group
                              max_lr = lr, # Upper learning rate boundaries in the cycle for each parameter group
                              step_size_up = 40,
                              step_size_down=1000,# Number of training iterations in the increasing half of a cycle
                              mode = "exp_range",
                              gamma=0.8),
                Cos_restart=lambda optimizer : torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                T_0 = 800,# Number of iterations for the first restart
                                                T_mult = 1, # A factor increases TiTi​ after a restart
                                                eta_min = 1e-8), # Minimum learning rate
                Exp_LR= lambda optimizer :torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5),
                )