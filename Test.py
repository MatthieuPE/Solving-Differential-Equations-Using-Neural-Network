import torch
import numpy as np
nn = torch.nn
import matplotlib.pyplot as plt
from Class_EQD_torch_O1_O2 import * # import all the file
from Class_Inverse_Function import Inverse_fct
import inspect
import re
import numba as nb

   
"""

=========================  TEST OF EQD OF 1ST ORDER ========================== 

"""


# ============== List of equations and their analytical solution ==============

# ================================= Order 1 ================================= "

dic_eq1={"y'+y":[lambda u,up,x:u+up,lambda x,x0,y0:y0*np.exp(x0-x)],
        "y'-0.5":[lambda u,up,x:up-0.5,lambda x, x0,y0:0.5*(x-x0)+y0],
        "y'-y/x":[lambda u,up,x:up-u/x,lambda x,x0,y0:y0*x/x0],
        "y'+y^2":[lambda u,up,x: up+u**2,lambda x,x0,y0:1/(x+(1-y0*x0)/y0)],
        "y'+exp(x-y)":[lambda u,up,x:up+torch.exp(x-u),lambda x,x0,y0:np.log(np.exp(y0)+np.exp(x0)-np.exp(x))],
        "xy'-y(1+ln(y)-ln(x)":[lambda u,up,x:x*up-u*(1+torch.log(u)-torch.log(x)),lambda x,x0,y0:x*np.exp(x/x0*np.log(y0/x0))],
        "y''+y":[lambda u,up,upp,x:upp+u,lambda x,x0,y0,x0p,y0p:x]}


if 'eqd' not in dir():
    
    " List of equations that can be tested "   
    
    " === Ordre 1 === "
    
    #eqd=EqDiff1(dic_eq1["y'+y"][0],build1DFunc(15,3) , boundc = [ (2,3),],xvals=torch.arange(1.5,2.5,0.001))
    
    #eqd=EqDiff1(dic_eq1["y'-0.5"][0],build1DFunc(15,3) , boundc = [ (2,3),],xvals=torch.arange(1.5,2.5,0.001))
    
    # This one diverges for x0=0 and x=0
    #eqd=EqDiff1(dic_eq1["y'-y/x"][0],build1DFunc(15,3) , boundc = [ (2,3),],xvals=torch.arange(1.5,2.5,0.001))
    
    # Solution diverges if we study in the interval where the is x=(y0xO-1)/y0
    #eqd=EqDiff1(dic_eq1["y'+y^2"][0],build1DFunc(15,3) , boundc = [ (2,3),],xvals=torch.arange(2,3,0.001))
    
    # This one is defined from -inf to ln(exp(y0)+exp(x0))
    #eqd=EqDiff1(dic_eq1["y'+exp(x-y)"][0],build1DFunc(15,3) , boundc = [ (2,3),],xvals=torch.arange(1.5,2.5,0.001))
    
    # This one is defined from 0 (exclude) to +inf, and y0 needs to be >0. 
    # ATTENTION : it needs to add a activation function "softplus" on the last layer
    #eqd=EqDiff1(dic_eq1["xy'-y(1+ln(y)-ln(x)"][0],build1DFunc(15,3) , boundc = [ (2,3),],xvals=torch.arange(1.5,2.5,0.001))

    """
    Command to fit the NN to the choosen equation
    eqd.fit(opti('nadam',0.0001),epochs=100)
    
    Example of command to plot a model, the derivate, and the analytical solution
    eqd.plotModel2(dic_eq1["y'-y/x"][1])
    """
    
    pass


"""

=========================  TEST OF EQD OF 2ND ORDER ========================== 

"""


# Constants of "y''+y,bc" using 2 CI on the function (x0,y0),(x1,y1)
A = lambda c0, s0, y0, c1, s1, y1: y0 / c0 - s0/c0 * ((c0 * y1 - y0 * c1) / (s1 * c0 - s0 * c1))
B=lambda c0,s0,y0,c1,s1,y1: (c0*y1-y0*c1)/(s1*c0-s0*c1)

# Constants of "y''+y,bc" using CI on the function (x0,y0) and derivate (x1,y1p)
Ad = lambda c0, s0, y0, c1, s1, y1p: y0 / c0 - (s0/c0) * (c0*y1p+s1*y0)/(c1*c0+s0*s1)
Bd=lambda c0,s0,y0,c1,s1,y1p: (c0*y1p+s1*y0)/(c1*c0+s0*s1)


# List of differential equations of 2nd order with analytical solution for chosen CI
dic_eq2={"y''+y,bc":[lambda u,up,upp,x:upp+u,lambda x,x0,y0:A(torch.cos(x0[0]),torch.sin(x0[0]),y0[0],torch.cos(x0[1]),torch.sin(x0[1]),y0[1])*torch.cos(x)
                     +B(torch.cos(x0[0]),torch.sin(x0[0]),y0[0],torch.cos(x0[1]),torch.sin(x0[1]),y0[1])*torch.sin(x)],
        "y''+y,bcd":[lambda u,up,upp,x:upp+u,lambda x,x0,y0:Ad(torch.cos(x0[0]),torch.sin(x0[0]),y0[0],torch.cos(x0[1]),torch.sin(x0[1]),y0[1])*torch.cos(x)
                             +Bd(torch.cos(x0[0]),torch.sin(x0[0]),y0[0],torch.cos(x0[1]),torch.sin(x0[1]),y0[1])*torch.sin(x)],
        "y''-2y-2y^3,0.0.0.1":[lambda u,up,upp,x:upp-2*u-2*u**3,lambda x,x0,y0:torch.tan(x)],
        "y''-sin(y)":[lambda u,up,upp,x:upp-torch.sin(u),lambda x,x0,y0:2*torch.arccos(torch.tanh(torch.arctanh(1/torch.sqrt(torch.tensor(2.)))-x))]}


if 'eqd2' not in dir():
    " === Ordre 2 === "
    
    # Solution following y(x) = A cos(x)+B sin(x)
    # To fit using 2 CI on the function
    #eqd2=EqDiff2(dic_eq2["y''+y,bc"][0],build1DFunc(25,3) , boundc = [ (0,1),(torch.pi/2,0)],xvals=torch.arange(-0.5,3,0.001))
    
    # To fit using  CI on the function and the derivate
    #eqd2=EqDiff2(dic_eq2["y''+y,bcd"][0],build1DFunc(40,3) , boundc = [ (0,1),],boundcd=[(0,0),],xvals=torch.arange(-0.5,3,0.001))


    # ATTENTION: The analytical solution has been taken for y(0)=0 and y'(0)=1.
    # x has to contain 0, and to be between \pm pi/2. It diverges close to +-pi/2.
    #eqd2=EqDiff2(dic_eq2["y''-2y-2y^3,0.0.0.1"][0],build1DFunc(40,10) , boundc = [ (0,0),],boundcd=[(0,1),],xvals=torch.arange(-1.4,1.4,0.001))


    # ATTENTION: analytical solution only for y(0)=pi/2, y'(0)=sqrt(2)
    #eqd2=EqDiff2(dic_eq2["y''-sin(y)"][0],build1DFunc(40,3) , boundc = [ (0,torch.pi/2),],boundcd=[(0,torch.sqrt(torch.tensor(2.))),],xvals=torch.arange(-2.5,2.5,0.001))


    """
    Command to plot the model, the derivate and the analytical solution
    eqd2.plotModel2(dic_eq2["y''+y,bc"][1],XBC="x_bc_all",YBC="y_bc_all")
    """
    
    pass



"""

=======================  TEST OF CALIBRATION FUNCTION  ======================= 

"""


# --- Constant values of the problem ---

Enorm=5e3 # (GeV) used to normalize data


" === Energy response values === "
alpha=0.95 # value of the response for high energies

x0=20/Enorm # Energy where we know a value of R for low energies
rx0=0.8 # value of r at the point x0.


" === For r = alpha(1-1/(beta+E)^9) === "

p=9 # exponant in beta
beta=(1/(1-rx0/alpha))**(1/p)-x0

" === For r = alpha + b/(c+x) === "

a=0.05 # a= b+c
c=(x0*(rx0-alpha)-a)/(-1+alpha-rx0)
b=a-c

" === CI === "

xinf1=950/Enorm #value of energy that we choose to be equivalent to infinity
xinf2=990/Enorm
Rinf1=alpha*(1-1/(beta+xinf1)**p)
Rinf2=alpha*(1-1/(beta+xinf2)**p)

" ==== SIGMA ===="

sigma=0.1 # if it is a constant

# For sigma as a function
Sigma0=0.3
SigmaA=0.1
SigmaB=(Sigma0-SigmaA)*x0
SigmaB=4/Enorm #to limit calculation errors
SigmaC=(0.6-0.3)/(4500-1200)*Enorm


def generateResponses(muF, sigmaF, etrue, nresp):
    """Given functions for mu and sigma and a list of true energies (etrue),
    draw nresp responses from  a normal distribution (muF(etrue), sigmaF(etrue))

    returns an array of shape (etrue.size, nresp). 
    
    """
    mus = muF(etrue)
    sigmas = sigmaF(etrue)
    
    resp = np.random.normal(mus,sigmas,size=(nresp, etrue.size) ).T

    # now resp[i] are the responses for true energy etrue[i]
    
    return resp


# Examples of response functions

" +++ Energy responses +++ "

b_torch=torch.tensor((rx0-alpha)*x0)
b_nump=(rx0-alpha)*x0
def R_b(x):
    return  alpha + b_torch/x

def R_cte(x):
    return  alpha

def R_p(x):
    p=9 # exponant in beta
    beta=(1/(1-rx0/alpha))**(1/p)-x0
    return  alpha*(1-1/(beta+x)**p)

def R_a(x):
    a=0.05 # a= b+c
    c=(x0*(rx0-alpha)-a)/(-1+alpha-rx0)
    b=a-c
    return alpha + b/(c+x)

" +++ Mass reponse +++ "
coeff=(1.5-0.8)/(4500-500)*Enorm
coeff=1.75e-4*Enorm
B_nump=1.5-coeff*4500/Enorm
B_torch=torch.tensor(B_nump)
def R_m(x):
    # output's type as the input ones.
    if not isinstance(x, torch.Tensor):
        return coeff*x+B_nump
    else:
        return coeff*x+B_torch 

x_id=(1-B_nump)/coeff


# To use data in lorarithm (NOT TESTED)
# Change of variable from x to z
def z_norm(x,Lnorm=0): # Normalize the log.
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    z=torch.log(x*Enorm)
    if Lnorm==0:
        Lnorm=torch.max(z)
    z=z/Lnorm
    print(torch.max(z))
    return z,Lnorm

def z_log(x,Lnorm=0): # Take the log of normalized energy
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    z=torch.log(x)
    Lnorm=1
    return z,Lnorm


class Analyse:
    
    """
    To obtain the solution of differential equation on calibration function,
    and manipulate the results.
    """
    
    def __init__(self,R,eq='u',z=z_norm):
        self.R=R 
        self.unknown=eq
        
        if not os.path.exists("Figures"):
            os.makedirs("Figures")
            
        # sigma as a constant
        #self.SIGMA=lambda x:sigma
        
        # Sigma as a function
        # for energy
        self.SIGMA=lambda x: SigmaA*x+SigmaB
        
        self.eq=lambda u,up,upp,x:upp*(self.SIGMA(x))**2+up*(x-u*self.R(u)) 
        
        # Right BC for each response function
        if self.R.__name__=="R_b": 
            self.SIGMA=lambda x:sigma
            # CI diff
            self.u=EqDiff2(self.eq,build1DFunc(25,3) , boundc=[(xinf2,(xinf2-b_nump)/alpha)],boundcd=[(xinf1,1/alpha)],xvals=torch.cat((torch.arange(1e-6,0.2,0.00005),torch.arange(0.2,0.4,0.0001),torch.arange(0.4,1,0.001))))
            #CI ident
            #self.u=EqDiff2(self.eq,build1DFunc(25,3) , boundc=[(xinf2,(xinf2-b_nump)/alpha)],boundcd=[(xinf2,1/alpha)],xvals=torch.cat((torch.arange(1e-6,0.3,0.0001),torch.arange(0.3,1,0.001))))

        if self.R.__name__=='R_p' or  self.R.__name__=='R_a':
            # CI diff
            self.u=EqDiff2(self.eq,build1DFunc(25,3) , boundc=[(xinf2,xinf2/Rinf2),(0,0)],boundcd=[(xinf1,1/alpha)],xvals=torch.cat((torch.arange(1e-6,0.2,0.00005),torch.arange(0.2,0.4,0.0001),torch.arange(0.4,1,0.001))))
            #self.u=EqDiff2(self.eq,build1DFunc(25,3) , boundc=[(xinf2,xinf2/Rinf2),(0,0)],boundcd=[(xinf1,1/alpha)],xvals=torch.arange(1e-6,1,0.001))
            #CI ident
            #self.v=EqDiff2(self.eq,build1DFunc(25,3) , boundc=[(xinf2,xinf2/Rinf2)],boundcd=[(xinf1,1/alpha)],xvals=torch.cat((torch.arange(1e-6,0.3,0.0001),torch.arange(0.3,1,0.001))))
         
        if self.R.__name__=='R_m':
            # for mass
            self.SIGMA = lambda x : SigmaA*x+SigmaB+SigmaC*x**2
            
            # homogen data
            self.u=EqDiff2(self.eq,build1DFunc(25,3) , boundc=[(x_id,x_id),(0,0)],boundcd=[],xvals=torch.arange(1e-6,1,0.0001))
           
            # overfit at low values
            #self.u=EqDiff2(self.eq,build1DFunc(25,3) , boundc=[(x_id,x_id),(0,0)],boundcd=[],xvals=torch.cat((torch.arange(1e-6,0.6,0.00005),torch.arange(0.6,1,0.0001))))
            
            #overfit at the end and a little for low values
            #self.u=EqDiff2(self.eq,build1DFunc(25,3) , boundc=[(x_id,x_id),(0,0)],boundcd=[],xvals=torch.cat((torch.arange(1e-6,0.1,0.00005),torch.arange(0.1,0.7,0.0001),torch.arange(0.7,1,0.00005))))
            
            #overfit at very low vlaue and high values
            #self.u=EqDiff2(self.eq,build1DFunc(25,3) , boundc=[(x_id,x_id),(0,0)],boundcd=[],xvals=torch.cat((torch.arange(1e-6,0.01,0.000005),torch.arange(0.01,0.1,0.00005),torch.arange(0.1,0.7,0.0001),torch.arange(0.7,1,0.00005))))
            
            # Semi normalized data, for low and high energies
            #self.u=EqDiff2(self.eq,build1DFunc(25,3) , boundc=[(x_id,x_id),(0,0)],boundcd=[],xvals=torch.cat((torch.arange(1e-6,0.2,0.00005),torch.arange(0.2,0.4,0.0001),torch.arange(0.4,0.8,0.001),torch.arange(0.8,1.5,0.0001),torch.arange(1.5,2,0.001))))
            
        if self.unknown=='w': # to solve equation on w (with logaritmic data)
            self.z=z(self.u.xvals)
            self.Lnorm=self.z[1]
            self.eq=lambda u,up,upp,x:upp-up*(self.Lnorm+up/(self.SIGMA(x))**2*(u-self.R(torch.exp(x*self.Lnorm)/Enorm)))  
            self.w=EqDiff2(self.eq,build1DFunc(25,3) , [(z(xinf2,Lnorm=self.Lnorm),alpha*xinf2+(rx0-alpha)*x0)],boundcd=[(z(x0,Lnorm=self.Lnorm),alpha*self.Lnorm*x0)],xvals=self.z[0])
            print(torch.min(self.z[0]),torch.max(self.z[0]))
        
        if self.unknown=='v': # to solve equation on v (inverse function)
            self.eq=lambda u,up,upp,x:upp-up**2/(self.SIGMA(x))**2*(u-x*self.R(x)) 
            if self.R.__name__=="R_b":
                self.v=self.v=EqDiff2(self.eq,build1DFunc(25,3) ,  boundc=[(xinf2,alpha*xinf2+b_nump)],boundcd=[(x0,alpha)],xvals=self.u.xvals)
            if self.R.__name__=="R_p":
                self.v=self.v=EqDiff2(self.eq,build1DFunc(25,3) ,  boundc=[(xinf2,alpha*xinf2)],boundcd=[(xinf1,alpha)],xvals=self.u.xvals)
        
        
        if self.unknown=='a':# to solve equation on "a=u/x"
            self.eq=lambda u,up,upp,x:(self.SIGMA(x))**2*(x*upp+2*up)+(u+x*up)*(x-x*u*self.R(x*u))  
            if self.R.__name__=="R_p":
                self.a=EqDiff2(self.eq,build1DFunc(25,3) , boundc= [(xinf2,1/alpha)],boundcd=[(xinf2,0)],xvals=self.u.xvals)
            if self.R.__name__=="R_b":
                self.a=EqDiff2(self.eq,build1DFunc(25,3) , boundc= [(xinf2,(1-b_nump/xinf2)/alpha)],boundcd=[(xinf1,b_nump/(alpha*xinf1**2))],xvals=self.u.xvals)
        
        
    def load(self,file):
        """
        To load a NN model.
        Please pay attention to use the same Enorm as the model loaded.
        """
        config = torch.load(file)
        model = getattr(self,self.unknown).model
        model.load_state_dict(config)
        
        #set the best_loss value from the title of the file.
        pattern = r"\w+_(\d+\.\d+e[-+]?\d+)\.pth" 
        match = re.search(pattern, file)
        if match:
            extracted_number = match.group(1)
            getattr(self,self.unknown).best_loss=torch.tensor(float(extracted_number))


    def solve_eq(self,epochs=1000,learning_rate_max=1e-3,lr_adapt=False,Print=50,dynamic=False,evolution=False,batch_size=500):
        # to avoid divergence
        if self.unknown=='w' and learning_rate_max>0.01:
            learning_rate_max=0.001 
            print("Learning_rate_max has been reduced to 0.001")
        if self.unknown=='a' and learning_rate_max>0.01:
            learning_rate_max=0.001
            print("Learning_rate_max has been reduced to 0.001")
        
        # For the name of the files, verify the type of sigma
        if inspect.isfunction(self.SIGMA):
            is_constant = True
            constant_value = self.SIGMA(0)  
            for x in range(1, 10):  # Check if sigma changes for different x values
                if self.SIGMA(x) != constant_value:
                    is_constant = False
                    break
            if is_constant:
                sigma_type = constant_value
            else:
                sigma_type = "fonction"
        
        # train the model
        getattr(self,self.unknown).fit(opti('diffgrad'),epochs=epochs,batch_size=batch_size,Print=Print,lr=learning_rate_max,lr_adapt=lr_adapt,dynamic=dynamic,evolution=evolution,unknown=self.unknown,sigma=sigma_type,R=self.R.__name__)
        self.x=self.u.xvals.detach().reshape(-1,1)
        self.ux = self.u.model( self.x )
        if self.unknown=='w': # Change the variable from z to x
            self.x=torch.exp(self.x*self.Lnorm)/Enorm
            print(torch.max(self.x),torch.min(self.x))
        
        if self.unknown=='v': 
            # To prepare the data to inverse the function
            self.xtrue=getattr(self,self.unknown).xvals.detach().reshape(-1,1)
            self.vx=getattr(self,self.unknown).model(self.xtrue)
            
        if self.unknown=='a': 
            # Obtain the u value by multiplying by x.
            self.x=getattr(self,self.unknown).xvals.detach().reshape(-1,1)
            self.ux=self.x*getattr(self,self.unknown).model(self.x)
   
             
    def inverse(self,epochs=1000,learning_rate_max=1e-3):
        
        "To inverse a function using neural network"
        
        if not hasattr(self, 'xtrue') or not hasattr(self, 'vx'):
            raise Exception("Must run 'self.solve_eq()' to obtain a function to inverse")
        if not hasattr(self,'u'):
            self.u=Inverse_fct(build1DFunc(25,3),self.xtrue,self.vx) 
        self.u.fit(opti('diffgrad'),epochs=epochs,lr=learning_rate_max)
        self.x=self.u.fct_value.detach().reshape(-1,1)
        self.ux = self.u.model(self.x) # on le re calcule à partir des poids fixé 
       
    def calibrate(self,N=10000,e_true=None,ana=False,hist=False):
        """ 
        To generate gaussian data, calibrate them using trained NN,
        plot the calibrated response to check if the calibration is correct 
        """
        if e_true==None:
            # Take the training data
            self.e_true = self.u.xvals.numpy()
        else:
            self.e_true=e_true.numpy()
            
        # Generate gaussian of sigma=sigma/x centered in R(e_true), with N MC generations
        r_reco = generateResponses(self.R, lambda x: self.SIGMA(x)/x, self.e_true, nresp=N)
        
        self.R_reco = r_reco.mean(axis=1) 
        
        self.e_reco = np.multiply(r_reco, self.e_true[:, np.newaxis])
        
        ec_mean = []
        ec_max=[]
        ec_meanANA= []
        fontsize=14 
        for row in self.e_reco: #for each gaussian
            x = torch.from_numpy(row).reshape(-1, 1).to(self.u.model[0].weight.dtype)
            self.e_calib = self.u.model(x) # calibrate the energies of the gaussian
            if ana: #if we study the eqd with analytic solution
                #e_calibth=(x-b_nump)/alpha
                e_calibth=x/alpha
                ec_meanANA.append(e_calibth.mean().item())
            if self.unknown=='a':
                # need to multiply by x
                self.e_calib = x*self.a.model(x)
            
            # find the maximum of the calibrated distribution using maximum
            Bin=int(np.sqrt(N)) # number of bins in the histogram
            max_bin_center,max_poly=mode_dist(self.e_calib.detach().numpy().flatten(),Bin)
            ec_max.append(max_bin_center)
            
            # or using the mean
            ec_mean.append(self.e_calib.mean().item())
            
        # Calculate the calibrated response
        self.R_calib = np.array(ec_mean) / self.e_true
        self.R_calib_max = np.array(ec_max) / self.e_true
         
        
        # plot the results
        fig, ax = plt.subplots()
        ax.axhline(y=1, color='r', linestyle='dashed',label='R=1')
        if ana:
            self.R_calibANA=np.array(ec_meanANA)/self.e_true
            ax.plot(self.e_true*Enorm, self.R_calibANA, label="Response u(x)=x/alpha")
        ax.plot(self.e_true*Enorm, self.R_calib_max, "b:", label="Response using max")
        ax.plot(self.e_true*Enorm, self.R_calib, "k:", label="Response using mean")
        ax.set_ylabel("Calibrated response",fontsize=fontsize)  
        ax.set_xlabel("Energy (GeV)",fontsize=fontsize)  
      
        # Set the X- axis
        xmin, xmax = np.min(self.e_true*Enorm).item(), np.max(self.e_true*Enorm).item()
        major_locator_x = MultipleLocator(base=(xmax - xmin) / 5)  
        minor_locator_x = MultipleLocator(base=(xmax - xmin) / 20)  
        ax.xaxis.set_major_locator(major_locator_x)
        ax.xaxis.set_minor_locator(minor_locator_x)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_ticks_position('both')
        ax.set_xlim(xmin, xmax)
        
        # Set the Y - axis
        ymin1, ymax1 = np.min(self.R_calib).item(), np.max(self.R_calib).item()
        ymin2, ymax2 = np.min(self.R_calib_max).item(), np.max(self.R_calib_max).item()
        ymin, ymax = np.min([ymin1, ymin2]), np.max([ymax1, ymax2])
        ymax+=ymax/25 # to set a step between max and limit of the plot.
        ymin-=ymin/25
        major_locator_y = MultipleLocator(base=(ymax - ymin) / 5)  
        minor_locator_y = MultipleLocator(base=(ymax - ymin) / 20)
        ax.yaxis.set_major_locator(major_locator_y)
        ax.yaxis.set_minor_locator(minor_locator_y)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.yaxis.set_ticks_position('both')
        ax.set_ylim(ymin, ymax)
            
        ax.legend(fontsize=12) 
         
        # Save figure
        folder = "Figures"
        filename = os.path.join(folder, f"{self.R.__name__}_Calibration_Enorm{Enorm}_bound{int(xmin)}_{int(xmax)}")
        plt.savefig(filename + ".pdf", format="pdf")
        plt.savefig(filename + ".png", format="png")
        plt.show()
    
    def plot_hist(self, e_true=0.9, N=10000):
        """
        To plot the distribution for a chosen etrue, before and after calibration.
        """
        
        Bin=int(np.sqrt(N)) # number of bins in the histogram
        e_true = np.array([e_true])  
        # Generate gaussian measurement
        r_reco = generateResponses(self.R, lambda x: self.SIGMA(x)/x, e_true, nresp=N)
        e_reco = np.multiply(r_reco, e_true)
     
        x = torch.from_numpy(e_reco).reshape(-1, 1).to(self.u.model[0].weight.dtype)
        e_calib = self.u.model(x).detach().numpy().flatten()
     
        mean_e_calib = np.mean(e_calib) #mean of the calibrated distribution
       
        max_bin_center,max_poly=mode_dist(e_calib,Bin) #max of the calibrated distribution
        
        # plot settings
        fig, ax = plt.subplots()
        ax.hist(e_reco.flatten()*Enorm, Bin, alpha=0.5,label=r'$E_r$')
        ax.hist(e_calib*Enorm, Bin, rwidth=1, alpha=0.5,label=r'$E_c$')
        ax.axvline(x=e_true*Enorm, color='b', label='Theoritical e_true ')
        ax.axvline(x=mean_e_calib*Enorm, color='r', linestyle='--', label='e_true with NN + mean')
        ax.axvline(x=max_bin_center*Enorm, color='g', linestyle='--', label='e_true with NN + max')
        plt.legend(fontsize=10)
        ax.set_ylabel('Counts',fontsize=14)
        ax.set_xlabel('Energy (GeV)',fontsize=14)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        
        folder = "Figures"
        filename = os.path.join(folder, f"{self.R.__name__}_distribution_Enorm{Enorm}_etrue{e_true}")
        plt.savefig(filename + ".pdf", format="pdf")
        plt.savefig(filename + ".png", format="png")
        plt.show()
        print("With the mean of e_calib, Rcalib = ",mean_e_calib/e_true,". With the max of bins, R_calib = ",max_bin_center/e_true,". With the fit, R_calib = ",max_poly/e_true)

    def compare_analytic(self,func,data='u',Print_functions=False,x=None):
        " To plot |theory / function from NN| "
        if not x==None:
            x=x.detach().reshape(-1, 1)
            F=func(x.numpy())
            pred = self.u.model(x).detach().numpy()
        else:
            if data=='v':
                x=self.xtrue.detach().numpy()
                F=func(x) # value of theoritical function
                pred=self.vx.detach().numpy()
            if data=='u':
                x=self.x.detach().numpy()
                F=func(x)
                pred=self.ux.detach().numpy()
        
        Diff=abs(F/pred)
        x=(Enorm*x).numpy()
        fig, ax = plt.subplots()
        if Print_functions:
            # if we want to plot the functions
            ax.plot(x,pred,label='prediction')
            ax.plot(x,F,label='Analytical')
        ax.axhline(y=1, color='r', linestyle='dashed',label='Ratio=1')
        ax.plot(x,Diff,'b',label='|Linear approximation/predictions|')
        ax.set_ylabel('|Ratio|',fontsize=14)
        ax.set_xlabel('Energy (GeV)',fontsize=14)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        xmin,xmax=np.min(x),np.max(x)
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=12)
        
        folder = "Figures"
        filename = os.path.join(folder, f"{self.R.__name__}_Ratio_Enorm{Enorm}_bound{int(xmin)}_{int(xmax)}")
        plt.savefig(filename + ".pdf", format="pdf")
        plt.savefig(filename + ".png", format="png")
        plt.show()
    
    def plot_width(self,e_true=None,N=10000):
        """
        To plot the width of the distribution before and after the calibration.
        We consider the width as the one containing 68% of the data, 
        with 16% to the left and 16% to the right.
        """
        if e_true==None:
            # Take the training data
            self.e_true = self.u.xvals.numpy()
        else:
            self.e_true=e_true.numpy()
            
        # Generate gaussian of sigma centered in R(e_true), with N MC generations
        r_reco = generateResponses(self.R, lambda x: self.SIGMA(x)/x, self.e_true, nresp=N)
        
        self.R_reco = r_reco.mean(axis=1) 
        
        self.e_reco = np.multiply(r_reco, self.e_true[:, np.newaxis])
        
        WIDTH=[]
        WIDTH_uncal=[]
        fontsize=14
 
        for row in self.e_reco: #for each gaussian
            # Calculate the width before calibration
            x = torch.from_numpy(row).reshape(-1, 1).to(self.u.model[0].weight.dtype)
            Bin=int(np.sqrt(N))
            Hist,edges=np.histogram(row.flatten(),bins=Bin)
            centers = 0.5 * (edges[1:] + edges[:-1])
            width=get_width(Hist,centers)
            WIDTH_uncal.append(width)
            
            
            self.e_calib = self.u.model(x) # calibrate the energies of the gaussian                
            if self.unknown=='a':
                # need to multiply by x
                self.e_calib = x*self.a.model(x)
            # calcutate the width after calibration
            Bin=int(np.sqrt(N))
            Hist,edges=np.histogram(self.e_calib.detach().numpy().flatten(),bins=Bin)
            centers = 0.5 * (edges[1:] + edges[:-1])
            width=get_width(Hist,centers)
            WIDTH.append(width)
         
        self.width=WIDTH
        self.width_uncal=WIDTH_uncal
        
        # Plot settings
        fig, ax = plt.subplots()
        ax.plot(self.e_true*Enorm, np.array(self.width)*Enorm, "k:", label="width after calibration")
        ax.plot(self.e_true*Enorm, 2*self.SIGMA(self.e_true)*Enorm, "brown", label="width before calibration (theory)")
        ax.plot(self.e_true*Enorm, np.array(self.width_uncal)*Enorm, "b--", label="width before calibration")
        ax.set_ylabel("Width (GeV)",fontsize=fontsize)
        ax.set_xlabel(r'$E_t(GeV)$',fontsize=fontsize)
        
        # Set the X- axis
        xmin, xmax = np.min(self.e_true*Enorm).item(), np.max(self.e_true*Enorm).item()
        major_locator_x = MultipleLocator(base=(xmax - xmin) / 5) 
        minor_locator_x = MultipleLocator(base=(xmax - xmin) / 20)  
        ax.xaxis.set_major_locator(major_locator_x)
        ax.xaxis.set_minor_locator(minor_locator_x)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.xaxis.set_ticks_position('both')
        ax.set_xlim(xmin, xmax)
        
        # Set the Y - axis
        ymin1, ymax1 = np.min(np.array(self.width)*Enorm).item(), np.max(np.array(self.width)*Enorm).item()
        ymin2, ymax2 = np.min(self.SIGMA(self.e_true)*Enorm).item(), np.max(self.SIGMA(self.e_true)*Enorm).item()
        ymin, ymax = np.min([ymin1, ymin2]), np.max([ymax1, ymax2])
        ymax+=ymax/25 # to set a step between max and limit of the plot.
        ymin-=ymin/25
        major_locator_y = MultipleLocator(base=(ymax - ymin) / 5)  
        minor_locator_y = MultipleLocator(base=(ymax - ymin) / 20) 
        ax.yaxis.set_major_locator(major_locator_y)
        ax.yaxis.set_minor_locator(minor_locator_y)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_ticks_position('both')
        ax.set_ylim(ymin, ymax)
        
        ax.legend(fontsize=12) 
        
        # Save figure
        folder = "Figures"
        filename = os.path.join(folder, f"{self.R.__name__}_width_Enorm{Enorm}_bound{int(xmin)}_{int(xmax)}")
        plt.savefig(filename + ".pdf", format="pdf")
        plt.savefig(filename + ".png", format="png")
        plt.show()
        
        

def get_width(hcontent,centers):
    """"
    Calculate the width of a distribution.
    Take the value of the histogram, and the centers of the bins in input.
    """
    def quantiles(hcontent, qList, returnCS=False):
        cs=np.cumsum(hcontent,dtype=np.float32)
        cs/=cs[-1]
        r = np.searchsorted(cs, qList )
        if returnCS:
            return r, cs
        return r
    qIndices = quantiles(hcontent, [0.16,0.84] )
    
    qi1, qi2 = qIndices
    
    iqr = centers[qi2] - centers[qi1]
    return iqr


@nb.njit #optimized with numba. 
def mode_dist(dist,Bin):
    """
    Return the most probable value of a distribution.
    Inputs : values and number of bins.
    """
    hist,edges=np.histogram(dist,bins=Bin)
    Tot=np.sum(hist)
    centers = 0.5 * (edges[1:] + edges[:-1])
    max_dist=np.max(hist)
    max_bin_index = np.argmax(hist) # bin the most filled
    port=0
    left=max_bin_index
    right=max_bin_index
    target=30/100
    while port/Tot<target:
        # Find the interval around the max containing at least 30% of values
        left-=1
        right+=1
        port=np.sum(hist[left:right])
    
    # Define the maximum using weighted sum.
    max_bin_center= np.sum(hist[left:right]*centers[left:right])/np.sum(hist[left:right])
    
    # Find the max using a polynomial fit : not working well.
    # rangemin=centers[left]
    # rangemax=centers[right]
    # bin_left = edges[max_bin_index] 
    # bin_right = edges[max_bin_index + 1]
    # hist,edges=np.histogram(dist,bins=int(Bin*target),range=(rangemin,rangemax))
    # centers = 0.5 * (edges[1:] + edges[:-1])
    # max_poly=extremaPoly2( hist, 0.5*(centers[0]+centers[-1]), bin_right-bin_left)[0]
    max_poly=1.0
    
    print(centers[left],max_bin_center,centers[right])
    return max_bin_center, max_poly


# Fit to distribution
_sgcoeff2 = {}
def fitPoly2Matrix(ny):
    a = _sgcoeff2.get(ny,None)
    if a is None:
        from scipy.signal import savgol_coeffs
        a = np.matrix( np.zeros(shape=(3,ny)))
        a[0,:] = savgol_coeffs(ny,2)
        a[1,:] = -savgol_coeffs(ny,2,1)
        a[2,:] = 0.5*savgol_coeffs(ny,2,2)
        _sgcoeff2[ny] = a
    return a

def extremaPoly2(y,x0, xwidth, y_err=None):
    coeffs = fitPoly2Matrix(len(y))
    a = coeffs.dot( y ).A[0]

    oneover2a = 0.5/a[2] 
    m = -a[1]*oneover2a
    x = m*xwidth+x0

    
    
    if y_err is None: y_err=np.sqrt(y)
    err2 =  ((coeffs.A*y_err)**2).sum(1)

    x_err2 = oneover2a*oneover2a*err2[1] + err2[2]*(a[1]*oneover2a*2*oneover2a)**2 
    
    return x, np.sqrt(x_err2)*xwidth

        
"""

=======================  FUNCTION INVERSION CLASS TEST  ======================= 

"""


# =============== Analytical inverse function that we can test ===============


dic_fctinv={"linear":[lambda x:5*x,lambda x:x/5],
            "exponential":[lambda x:torch.exp(x),lambda x:torch.log(x)],
            "cos":[lambda x:torch.cos(x),lambda x:torch.arccos(x)]}

XTRUE=torch.arange(0,1,0.001)

# linear
#inv=Inverse_fct(build1DFunc(20,3),XTRUE,dic_fctinv["linear"][0](XTRUE).detach())

# exponential
#inv=Inverse_fct(build1DFunc(20,3),XTRUE,dic_fctinv["exponential"][0](XTRUE).detach())

# cos
#inv=Inverse_fct(build1DFunc(20,3),XTRUE,dic_fctinv["cos"][0](XTRUE).detach())

