import torch
import numpy as np
nn = torch.nn
import matplotlib.pyplot as plt
import torch_optimizer as extoptim
import glob
import os
import copy
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
plt.ion() 


def build1DFunc( lsize, nlayer=3, activ=nn.Mish ):
    """Returns a torch model acting as simple 1D function.
    The model is a MLP with 'nlayer' layers each of size 'lsize'
    """
    # take a scalar in input, apply linear function, then the activation function Mish
    seq = [ nn.Linear( 1, lsize  ), nn.Mish()]
    for i in range(nlayer-2):
        # Layer of lsize neurons linked to an layer with lsize neurons
        seq+= [ nn.Linear( lsize, lsize , ),nn.Mish(),]    
    # Last layer take lsize input and return a scalar.
    seq+= [ nn.Linear( lsize, 1 )]
    m= nn.Sequential(*seq)
    return m

    
class EqDiff1:
    """
    This class encapsulates all the information needed to solve a 1st order differential equation.

    It makes use of 3 ingedient
     * a function of (y,dydx,x) defining the differential equation. Ex : if the equation is "y+dy/dx = 0" then the function will be : lambda y,dydz,x : y+dyddx
     * a torch Module to model the solution of the equation.
     * boundary condition to constrain to 1 specific solution. It is given in the form (y0,x0) and will be enforce as a quadratic loss term. 
    """
    
    def __init__(self, eqExpr, m, boundc=[], xvals=None):
        """
        eqExpr : function of (y,dydx,x) returning the expression of the differential equation
        m : a torch.Module to model the solution (must take a 1D input and return a 1D output)
        boundc : tuples in the form [(x0,y0),...] to enforce boundary conditions. In principle exactly 1 is needed to enforce the solution of a 1st order diff eq.
        xvals : a range of x values representing the domain on which we solve the equation. defaults to 2000 points in [-1,1]
        """
        self.model = m
        self.setEq(eqExpr)
        self.setBoundC(boundc)
        
        if xvals is None:
            # data to train the neural network
            xvals = torch.arange(-1,1,0.001)
            
        self.xvals = xvals
        self.totalEpochs = 0

    def setBoundC(self, boundc):
            
        self.boundc = boundc
        
        # stock the boundaries conditions (bc) in colomn vectors
        self.x_bc = torch.tensor( [ x0 for (x0,v0) in boundc],dtype=torch.float32).reshape(-1,1)
        self.y_bc = torch.tensor( [ v0 for (x0,v0) in boundc],dtype=torch.float32).reshape(-1,1)
        
        def bcLoss(model):
            # Define the loss function on the bc.
            return (model(self.x_bc) - self.y_bc)**2
        self.bcLoss =bcLoss


    def setEq(self, eqExpr):
        self.eqExpr=eqExpr
        def eqLoss(u, up,x):
            # Define the loss function on the differential equation
            return eqExpr(u,up,x)**2
        self.eqLoss = eqLoss
        
    def save_config(self,optimizer,loss_value,lr_adapt,unknown='u',sigma=0.1,R='R_p'):
        " To save a chosen configuration of the model "
        
        optimizer_name = type(optimizer).__name__                 
        # Create a directory if it doesn't exist
        if not os.path.exists("configs"):
            os.makedirs("configs")
                
            
        # Find in the directory files with same reponse, optimizer name, bc, if there is a scheduler, the type of sigma used, as the current model
        # and delete these files.
        boundc_str = "_".join([str(bc) for bc in self.boundc])
        previous_file =f"configs/{R}_{unknown}_{optimizer_name}_{boundc_str}_Scheduler.{lr_adapt}_Sigma.{sigma}_*.pth"
        previous_files = glob.glob(previous_file)
        for file in previous_files:
             os.remove(file)
        
        
        # Create the file name with these informations and the loss value
        file_name = f"configs/{R}_{unknown}_{optimizer_name}_{boundc_str}_Scheduler.{lr_adapt}_Sigma.{sigma}_{loss_value:.3e}.pth"
    
        # Save the model state dict to the file
        torch.save(self.model.state_dict(), file_name)
        print("Model saved")
        
    
    def fit(self, getOptim,batch_size=500,epochs=50, boundc=None, xvals=None, lr=1e-3, lr_adapt=False,evolution=False,dynamic=False,Print=1,unknown='u',sigma=0.1,R='R_p'):
        " To train the model on the choosen equation "
        
        if boundc is not None:
            self.setBoundC(boundc)
        if xvals is None:
            xvals = self.xvals
        
        # Create the optimizer for the model.
        optimizer = getOptim(self.model,lr)
        
        
        print("Parameters before the run : ")
        print("learning rate = ", optimizer.param_groups[-1]['lr'])
        showNorms(self.model) #print norms of the weights.
        
        
        if lr_adapt: # set scheduler if needed
            # Examples of existing scheduler to modify the learning rate.
            
            # scheduler = schedulerMap[ 'CyclicLR_triang' ](optimizer,lr)
            # scheduler = schedulerMap[ 'CyclicLR_triang2' ](optimizer,lr)
            # scheduler = schedulerMap[ 'CyclicLR_exp' ](optimizer,lr,gamma=0.8)
            scheduler = schedulerMap[ 'StepLR' ](optimizer)
            # scheduler = schedulerMap[ 'Cos_restart' ](optimizer)
            # scheduler = schedulerMap[ 'Exp_LR' ](optimizer)
            self.lr=scheduler.get_last_lr()
            
        # Initialize lists to store loss, gradients and learning rate values
        gradtot=[]
        losses = []
        learning_rates=[]
        if dynamic: # to plot evolution of loss, grad, lr, in dynamic
            if lr_adapt:
                fig, axs = plt.subplots(3)
            else:
                fig, axs = plt.subplots(2)
            axs[0].set_ylabel("Loss")
            axs[1].set_ylabel("Learning Rate")
            axs[1].set_xlabel("Epoch")
        
        nx = len(xvals)
        if batch_size>nx:
            batch_size=int(nx/2)  # to keep the batch smaller from the data size 
            print("The batch size has been modified to : ",batch_size)
        for epoch in range(epochs):
            grad=[]
            epoch_losses = []
            epoch_learning_rates = []
            
            # setup shuffled list of indices at which to pick  x values during training :
            xi = np.arange(nx).reshape( (nx,1) ) # create a list of indices from 1 to nx in column.
            np.random.shuffle(xi) # shuffle xi to avoid overfitting
            
            for start in range(0,xi.shape[0],batch_size):
                stop = start + batch_size

                # pick the x values at indices xi[start:stop]
                x_batch_train = xvals[torch.from_numpy(xi[start:stop])].reshape(-1,1)
                # calculate the loss
                loss_value, loss_eq, loss_bc = self.trainStep( x_batch_train, optimizer)
                optimizer.zero_grad()
                
                # Update the model weights
                loss_value.backward()
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
                    epoch_losses.append(loss_value.item())
                    if lr_adapt:
                        epoch_learning_rates.append(scheduler.get_last_lr()[0])

            if epoch %  Print == 0 and epoch != 0:   #print every Print epochs
                print(f"Training epoch : {epoch+self.totalEpochs}. Last losses: {loss_eq:.4e} + {loss_bc:.4e} = {loss_value:.4e}")
            
            if not hasattr(self, 'best_loss'):
                self.best_loss=loss_value # set the best loss value
            if loss_value<self.best_loss:
                # if we find a better loss value we store it
                self.best_loss=loss_value
                if self.best_loss < 1e-3: 
                    # start saving the model when loss < 1e-3
                    print(f"Model saved at epoch {epoch+self.totalEpochs}. Loss saved : {loss_eq:.4e} + {loss_bc:.4e} = {loss_value:.4e}")
                    self.save_config(optimizer,loss_value,str(lr_adapt),unknown,sigma=sigma,R=R)
                   
            if evolution:
                # to observe the evolution of the fitting.
                self.plotModelEvolution(epoch,epochs) 
            if lr_adapt:
                scheduler.step() # modify the learning rate
                self.lr=scheduler.get_last_lr()
            if dynamic:
                # Append the average loss, gradients and learning rate values for the epoch
                losses.append(sum(epoch_losses) / len(epoch_losses))
                gradtot.append( (sum(grad) / len(grad)) )
                start_epoch = max(0, epoch - 500) #to plot only 500 epochs
                axs[0].cla()  # Clear the plot
                axs[0].set_yscale('log')
                axs[0].plot(range(start_epoch, epoch + 1), losses[start_epoch:], 'b-')
                axs[1].cla()  
                axs[1].plot(range(start_epoch, epoch + 1), gradtot[start_epoch:], 'r-')
                if lr_adapt:
                    axs[2].cla()  # Clear the plot
                    learning_rates.append(sum(epoch_learning_rates) / len(epoch_learning_rates))
                    axs[2].plot(range(start_epoch, epoch + 1), learning_rates[start_epoch:], 'r-')
                plt.pause(0.01)
        self.totalEpochs += epochs
        
        print(f"Last loss at epoch {self.totalEpochs} : {loss_eq:.4e} + {loss_bc:.4e} = {loss_value:.4e}")
        print("Parameters after the run : ")
        print("learning rate = ", optimizer.param_groups[-1]['lr'])
        print(f" The best loss value obtained is: {self.best_loss.item():.4e}")
        showNorms(self.model)
        plt.ioff()
        plt.show()
        plt.close()


    def trainStep(self, x, optimizer,divergence=None):
        """Compute the derivative w.r.t inputs, then the diff equation & boundary conditions """
        # Split x from the current gradient  , to store gradient in another graph
        x = x.clone().detach().requires_grad_(True)
        # Calculate the value of the function : y(x)        
        y = self.model(x )
        optimizer.zero_grad()
        # Calculate the derivative of the function : (dy/dx)(x)
        dydx = torch.autograd.grad(y,x,torch.ones_like(y), create_graph=True)[0]
        # Calculate the loss on equation : 
        loss_eq =  self.eqLoss(y, dydx, x ).mean()
        # Calculate the loss on boundary condiction : 
        loss_bc = self.bcLoss( self.model ).mean()
        # Calculate the total loss
        loss_value = loss_eq  + loss_bc 
        
        return loss_value, loss_eq, loss_bc

    def evalDeriv(self, x):
        g=x.requires_grad
        x.requires_grad = True
        y = self.model(x)
        dydx = torch.autograd.grad(y,x,torch.ones_like(y), )[0]
        x.requires_grad = g
        return y, dydx
        
    def plotModel(self, otherF=None):
        x=self.xvals.detach().reshape(-1,1)
        y,dydx = self.evalDeriv(x) 
        plt.cla() #clear the current axes
        plt.plot(x, y.detach(), label='f') 
        plt.plot(x, dydx.detach(), label='fprime')
        if otherF is not None:
            plt.plot(x, otherF(x).detach(), label='reference')
        plt.legend()

    def plotModelEvolution(self,epoque,epoquetot,step=50):
        """
        To observe the evolution of the fitting.
        """
        i=int(epoque/step)
        transparence = np.linspace(0.1, 1, int(epoquetot/step))
        if epoque%step==0 and epoque!=0:#plot the function every 50 epochs
            x=self.xvals.detach().reshape(-1,1)
            y,dydx = self.evalDeriv(x) 
            plt.plot(x, y.detach(),"k",alpha=transparence[i]) 
        
    def plotModel2(self, otherF=None,XBC="x_bc",YBC="y_bc",Enorm=1,R='R_p'):
        """
        To plot the solution obtained with NN.
        Allows to plot theorytical function using the right BCs.
        """

        x = self.xvals.detach().reshape(-1, 1)
        y, dydx = self.evalDeriv(x)
        fig, ax = plt.subplots()
        E=x*Enorm # To plot using non-normalized energy.
        ax.plot(E, Enorm*y.detach(), label=r'$u(x)\times Enorm$')
        ax.plot(E, Enorm*dydx.detach(), label=r'$u^\prime(x)\times Enorm$')
        if otherF is not None:
            ax.plot(E, Enorm*otherF(x, getattr(self, XBC), getattr(self, YBC)).detach(), 'r--', label=r'$y=x$')
        
        # Plot settings:
            
        plt.legend(fontsize=12)
        
        # Set the X- axis
        xmin, xmax = torch.min(E).item(), torch.max(E).item()
        major_locator_x = MultipleLocator(base=(xmax - xmin) / 5)  
        minor_locator_x = MultipleLocator(base=(xmax - xmin) / 20)  
        ax.xaxis.set_major_locator(major_locator_x)
        ax.xaxis.set_minor_locator(minor_locator_x)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.xaxis.set_ticks_position('both')
        ax.set_xlim(xmin, xmax)  
        ax.set_xlabel(r'$E_r\, (GeV)$',fontsize=14)  
        
        # Set the Y - axis
        ymin, ymax = min(torch.min(y*Enorm).item(),torch.min(Enorm*dydx).item()), max(torch.max(Enorm*y).item(),torch.max(Enorm*dydx).item())
        ymax+=ymax/25 # to set a step between max and limit of the plot.
        major_locator_y = MultipleLocator(base=(ymax - ymin) / 5)
        minor_locator_y = MultipleLocator(base=(ymax - ymin) / 20)  
        ax.yaxis.set_major_locator(major_locator_y)
        ax.yaxis.set_minor_locator(minor_locator_y)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_ticks_position('both')
        
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(r'$E_c\, (GeV)$',fontsize=14)
        
        plt.show()
        
        # save plot
        folder = "Figures"
        filename = os.path.join(folder, f"{R}_Plot_Enorm{Enorm}_bound{int(xmin)}_{int(xmax)}")
        plt.savefig(filename + ".pdf", format="pdf")
        plt.savefig(filename + ".png", format="png")
        
        

# ============================== Order 2 ==================================


class EqDiff2(EqDiff1):
    """
    To solve an ODE of second order (linear of non linear), using NN.
    """
    def __init__(self,eqExpr, m, boundc=[],boundcd=[], xvals=None):

        if len(boundc+boundcd)<2:
            raise Exception("Must provide 2 boundary conditions")
        # we give to the object the same attributs and methods as the eqt of 1st order
        super().__init__(eqExpr, m, boundc, xvals) 
        self.setBoundc_deriv(boundcd) # initialize the bc on the derivate
        self.set_boundc_all(boundc, boundcd) # initialize the total bc.
       
    
    def set_boundc_all(self,boundc,boundcd):
        " To stock all the bc "
        self.boundc_all = copy.deepcopy(boundc)  #if the user gives bc only on the function
        if len(boundcd)>0:
            self.boundc_all+=boundcd # if the user gives bc also on the derivate
        self.x_bc_all = torch.tensor( [ x0 for (x0,v0) in self.boundc_all],dtype=torch.float32,requires_grad=True).reshape(-1,1)
        
        self.y_bc_all = torch.tensor( [ v0 for (x0,v0) in self.boundc_all],dtype=torch.float32).reshape(-1,1)
    
    
    def setBoundc_deriv(self,boundcd):
        " to store the bc on the derivate "
        self.boundcd=boundcd
        self.x_bcd = torch.tensor( [ x0 for (x0,v0) in boundcd],dtype=torch.float32,requires_grad=True).reshape(-1,1)
        self.yp_bcd = torch.tensor( [ v0 for (x0,v0) in boundcd],dtype=torch.float32).reshape(-1,1)
        
        def bcd_loss(model):
            " to calculate the loss function of the bc on the derivate "
            if len(self.boundcd)>0:
                # calculate y(x_0) 
                y0=model(self.x_bcd)
                # calculate y'(x_0) 
                y0p=torch.autograd.grad(y0,self.x_bcd,torch.ones_like(y0), create_graph=True)[0]
                return (y0p-self.yp_bcd)**2
            else:
                # if there is no bc on the derivate, thus the loss is 0
                return torch.zeros(1)
        self.bcdloss=bcd_loss
    
    def eqLosss(self,u, up,upp,x):
        " To calculate the loss function of the equation, but for order 2"
        return self.eqExpr(u,up,upp,x)**2
    
    def trainStep(self,x,optimizer):
        " To run the model and find the loss functions from the data "
        x = x.clone().detach().requires_grad_(True)
        # Calculate y(x)
        y=self.model(x)
        # Calculate y'(x)
        dydx = torch.autograd.grad(y,x,torch.ones_like(y), create_graph=True)[0]
        # Calculate y''(x)
        dydxdx=torch.autograd.grad(dydx,x,torch.ones_like(y), create_graph=True)[0]
        
        optimizer.zero_grad()
        
        # Calculate the mean of loss functions for equation, bc (function and derivate) of the data
        loss_eq =  self.eqLosss(y, dydx,dydxdx, x ).mean()
        loss_bc = self.bcLoss( self.model ).mean()
        loss_bcd=self.bcdloss( self.model ).mean()
        
        # Total loss function
        loss_value = loss_eq  + loss_bc + loss_bcd
        
        return loss_value, loss_eq, loss_bc+loss_bcd        
    
    
def showNorms(model):
    for (n,l) in model.named_modules(): # take each module l with its name n
        if isinstance(l, torch.nn.Linear):
            NN=torch.norm(l.weight).item() # calculate the norm of the matrix l. The ".item()"  gives the value of the tensor.
            N0=torch.norm(l.weight,dim=0).max().item()# Calculate the norm of the weights in each column, then take the max
            N1=torch.norm(l.weight,dim=1).max().item() # Calculate the norm of the weights in each lines, then take the max
            print(f'{n:15s} , {NN:.3f}, {N0:.3f}, {N1:.3f}')


def opti(name, l2=0,):
    """ short cut function to return an optimizer configured with given lr and l2 parameters """
    optiClass = optimizerMap[ name ]    
    opti = lambda model, scale_lr : optiClass(model.parameters(), lr= scale_lr, weight_decay=l2)
    # if name == 'adam':  # To use adam with amsgrad
    #    opti = lambda model, scale_lr : optiClass(model.parameters(), lr= scale_lr, weight_decay=l2,amsgrad=False)
    return opti
    

# A convenient directory of known optimizers            
optimizerMap= dict(
    adam = torch.optim.Adam,
    nadam = torch.optim.NAdam,
    adamw=torch.optim.AdamW,
    adamp = extoptim.AdamP,
    radam = torch.optim.RAdam,
    SGD=torch.optim.SGD,
    diffgrad = extoptim.DiffGrad,
    lamb = extoptim.Lamb,
    ranger=extoptim.Ranger
    )


# Directory of existing scheduler to modify learning rate.
schedulerMap=dict(
                StepLR=lambda optimizer : torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.82),
                CyclicLR_triang = lambda optimizer,lr: torch.optim.lr_scheduler.CyclicLR(optimizer, 
                              base_lr = 1e-12, # Initial learning rate which is the lower boundary in the cycle for each parameter group
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
                                                T_0 = 3000,# Number of iterations for the first restart
                                                T_mult = 1, # A factor increases TiTi​ after a restart
                                                eta_min = 1e-6), # Minimum learning rate
                Exp_LR= lambda optimizer :torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5),
                )