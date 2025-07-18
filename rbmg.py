import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import h5py
from plyer import notification
from scipy.optimize import linear_sum_assignment
import re

class RBM:
    
    def __init__(self, n_visible, n_hidden, dtype, batch_size, time, rdm, regu, T, filename, lr=0.01, gibbs_steps=10, epoch_max=1000):

        #Inicializo variables
        self.Nv = n_visible
        self.Nh = n_hidden
        self.dtype = dtype
        
        #Parámetros
        var_in = 1e-4
        self.var_in = var_in
        self.W = torch.randn(size=(self.Nh,self.Nv), dtype=self.dtype)*var_in
        self.b_v = torch.zeros(self.Nv, dtype=self.dtype)
        self.b_h = torch.zeros(self.Nh, dtype=self.dtype)
        self.sigma = torch.ones([self.Nh,self.Nv])
        self.A = 1/(2*self.sigma.mm(self.sigma.t()))-self.W.mm(self.W.t())/8
        #Training, hiperparámetros
        self.lr = lr
        self.emax = epoch_max
        self.gibbs_steps = gibbs_steps
        self.S_batch = batch_size
        
        self.rdm = rdm #True/False
        self.regu = regu #True/False
        self.regL2 = 0.002
        self.T = T
        self.time = time
        
        self.fname = filename
        
        self.mean = torch.zeros(1)
        self.std = torch.ones(1)
        self.hidden_std = torch.ones(1)
        self.ND = torch.distributions.normal.Normal(self.mean,self.std) #normal distribution
                
    def Energy(self,X):
        #Energía calculada a partir del dataset

        Xbv = -torch.sum(X.t()*self.b_v)
        WX_bh = - torch.sum(torch.log(1+torch.exp(self.W.mm(X).t()+self.b_h)))
        return Xbv+WX_bh

    def SetVisBias(self,D): #D = datos
        #inicialización para los bias de la capa visible
        #probabilidad limitada en rango
        
        NS = D.shape[1]
        prob1 = torch.sum(D,1)/NS
        prob1 = torch.clamp(prob1,min=1e-5)
        prob1 = torch.clamp(prob1,max=1-1e-5)
        self.b_v = -torch.log(1.0/prob1 - 1.0)

    def Energy_RBM(self,V,H):
        #Energía calculada a partir de los parámetros de la RBM
        
        I = - torch.sum(H*torch.tensordot(self.W,V,dims=1),0)
        F = - torch.tensordot(self.b_h,H,dims = 1) - torch.tensordot(self.b_v,V,dims = 1)
        return I + F

    def update(self,v_d,h_d,v_h,h_h,mh_d,mh_h):
        
        etaT = np.random.randn()

        if self.regu: # Regularización L2
            self.W += self.lr*(h_d.mm(v_d.t())-mh_h.mm(v_h.t()))/self.S_batch-self.regL2*self.W + torch.randn(self.Nh, self.Nv)*np.sqrt(2*self.lr*self.T)
        
        else:
            self.W += self.lr*(mh_d.mm(v_d.t())-mh_h.mm(v_h.t()))/self.S_batch + torch.randn(self.Nh, self.Nv)*np.sqrt(2*self.lr*self.T)
            
        self.b_v += self.lr*torch.mean((v_d-v_h),1) + torch.randn(self.Nv)*np.sqrt(2*self.lr*self.T)
        self.b_h += self.lr*torch.mean((mh_d-mh_h),1) + torch.randn(self.Nh)*np.sqrt(2*self.lr*self.T)
        #print(torch.mean(self.W),torch.mean(self.b_v),torch.mean(self.b_h))
    def Vsampling(self,H):
        mv = torch.sigmoid((self.W.t().mm(H).t()+self.b_v).t()) #probabilidad de los nodos visibles
	
        v = torch.bernoulli(mv) #estados de los nodos visibles
        return v,mv

    def Hsampling(self,V):
        
        h_mean = (self.W.mm(V).t()+self.b_h).t()
        mh = h_mean #sin sigmoide?
        h = torch.normal(mean=h_mean, std=torch.ones_like(h_mean))

        return h,mh

    def train(self,D): #introduzco dataset
            
        s = 0 #contador epochs
        m = 0 #contador minibatches
        
        emax = self.emax
        NB = int(D.shape[1]/self.S_batch)
        mcmc = self.gibbs_steps
        
        for t in range(emax): # bucle epochs
            loss = 0
            s += 1 #EP_TOT

            for k in range(NB): #bucle minibatches
                Xb = D[:,k*self.S_batch:(k+1)*self.S_batch] #minibatch
                v_d = Xb
                
                h_d,mh_d = self.Hsampling(v_d)

                if self.rdm: #Método RDM-k
                    Xr = torch.bernoulli(torch.rand(D.shape[0],self.S_batch))
                    v_h = Xr
                    for i in range(mcmc):
                        h_h,_ = self.Hsampling(v_h)
                        v_h,_ = self.Vsampling(h_h)
                        
                    # Xr = v_h
                    h_h, mh_h = self.Hsampling(v_h) 
 
                else: #Método PCD-k
                    v_h = Xb
                    
                    for i in range(mcmc):
                        h_h,_ = self.Hsampling(v_h)
                        v_h,_ = self.Vsampling(h_h)

                    h_h, mh_h = self.Hsampling(v_h)       
               
                self.update(v_d,h_d,v_h,h_h,mh_d,mh_h) #actualizo pesos
                A = 1/(2*self.sigma.mm(self.sigma.t()))-self.W.mm(self.W.t())/8
                #cor = torch.mm(v_d.t(), h_d) / self.S_batch  # <SiSj> para el minibatch actual
                
                if m in self.time:
                    f = h5py.File(self.fname,'a') #a - anexar
                    #print('Checkpoint...')
                    f.create_dataset('W'+str(m),data = self.W)
                    f.create_dataset('bias_v'+str(m),data = self.b_v)
                    f.create_dataset('bias_h'+str(m),data = self.b_h)
                    f.create_dataset('A'+str(m),data = A)
                    f.close()
                    
                m += 1
                loss +=torch.mean(torch.abs(v_d - v_h))
            print('It = ', t, 'loss',loss/s)

    def Sampling(self,X,it_mcmc=0): 
        
        if it_mcmc==0:
            it_mcmc = self.gibbs_steps

        v = X
        h,mh = self.Hsampling(v)
        v,mv = self.Vsampling(h)

        for t in range(it_mcmc-1):
            h,mh = self.Hsampling(v)
            v,mv = self.Vsampling(h)

        return v,mv,h,mh

    def ImConcat(self,X,ncol=10,nrow=5,sx=28,sy=28,ch=1): #Obtención de imágenes
        
        tile_X = []
        
        for c in range(nrow):
            L = torch.cat((tuple(X[i,:].reshape(sx,sy,ch) for i in np.arange(c*ncol,(c+1)*ncol))))
            tile_X.append(L)
            
        return torch.cat(tile_X,1)
    
    def PlotSampling(self,itmax):
        
        vinit = torch.bernoulli(0.5*torch.ones((self.Nv,50), dtype=self.dtype))
        vt = vinit

        Im = []
        Im.append(self.ImConcat(vt[:,:100].t(),ncol=1,nrow=50,sx=28,sy=28,ch=1))

        for i in range(itmax):
            ΔMC = 2**i
            vt,vis_m,_,_ = self.Sampling(vt,it_mcmc=2**i)
            
            if (i*1)*ΔMC == 10:
                Im.append(self.ImConcat(1-vis_m[:,:100].t(),ncol=1,nrow=50,sx=28,sy=28,ch=1))
            else:
                Im.append(self.ImConcat(vis_m[:,:100].t(),ncol=1,nrow=50,sx=28,sy=28,ch=1))
        
        AllIm=torch.cat(tuple(Im[i] for i in range(itmax)))

        plt.figure(dpi=300)
        ax = plt.gca()
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel('$\Delta$MC = $2^n$=(1,.., {}, ..., {}) '.format(2**(int(itmax/2)),2**(itmax)), rotation=270,fontsize=5)
        plt.title('epochs = {}, $\eta$ = {}, $N_h$ = {}, gibbs steps = {}, T = {}, RDM = {}, Reg = {} '.format(self.emax,self.lr, self.Nh, self.gibbs_steps, self.T, self.rdm, self.regu), fontsize = 6)
        plt.imshow(AllIm,cmap='gray')

    def PlotW(self):
    
        f = h5py.File(self.fname,'r')
        alltime = np.sort(f['alltime'])
        alls = []
        allt = []
        
        for t in alltime:
            ep = int(t) # epoch to which retrieve the RBM
            
            if not(('W'+str(ep)) in f): # check for last time
                break
        
            W = torch.tensor(f['W'+str(ep)])    
            _,s,_ = torch.svd(W)
            alls.append(s.reshape(s.shape[0],1))
            allt.append(ep)
        allt = np.array(allt)
        alls = torch.cat(tuple(alls),dim=1)

        plt.figure(dpi=150)
        plt.title('SVD Weights $\longrightarrow$epochs = {}, $\eta$ = {}, $N_h$ = {}, gibbs steps = {}, T= {}, RDM = {}, Reg = {} '.format(self.emax,self.lr, self.Nh, self.gibbs_steps, self.T, self.rdm, self.regu), fontsize = 6)
        plt.loglog(np.array(allt),alls.t());
        
    def PlotBias(self):
    
        f = h5py.File(self.fname,'r')
        alltime = np.sort(f['alltime'])
        allv = []
        allh = []
        allt = []
        
        for t in alltime:
            ep = int(t) # epoch to which retrieve the RBM
            
            if not(('bias_h'+str(ep)) in f): # check for last time
                break
        
            b_v = torch.tensor(f['bias_v'+str(ep)])    
            b_h = torch.tensor(f['bias_h'+str(ep)]) 

            allv.append(np.array(b_v))
            allh.append(np.array(b_h))
            allt.append(ep)
            
        allt = np.array(allt)
        #alls = torch.cat(tuple(alls),dim=1)

        plt.figure(dpi=150)
        plt.title('Visible bias $\longrightarrow$epochs = {}, $\eta$ = {}, $N_h$ = {}, gibbs steps = {}, T = {}, RDM = {}, Reg = {} '.format(self.emax,self.lr, self.Nh, self.gibbs_steps, self.T, self.rdm, self.regu), fontsize = 6)
        plt.loglog(np.array(allt),allv)

        plt.figure(dpi=150)
        plt.title('Hidden bias $\longrightarrow$epochs = {}, $\eta$ = {}, $N_h$ = {}, gibbs steps = {}, T = {}, RDM = {}, Reg = {} '.format(self.emax,self.lr, self.Nh, self.gibbs_steps, self.T, self.rdm, self.regu), fontsize = 6)
        plt.loglog(np.array(allt),allh)
        
    def PlotMSDW(self,perm):
    	#perm - true/false
    	
        plt.figure(dpi=150)
        f = h5py.File(self.fname,'r')

        timew = []
        for n in range (1,30):
            tw = 2**(2*n) #para coger un medio de los tw
            timew.append(tw)    
        timew = np.array(list(set(timew)))
        timew = np.sort(timew)  

        for ttw in timew:
            alls = []
            epw = int(ttw)
    
            if not(('W'+str(epw)) in f): # check for last time
                break    
            
            W_tw = torch.tensor(f['W'+str(epw)]) 
            W_tw = np.array(W_tw) #W(tw)
    
            alltime = []
            allt = []
            
            for m in range (1,30):
                tt = ttw + 2**m
                alltime.append(tt) 
                
            alltime = np.array(list(set(alltime)))
            alltime = np.sort(alltime)
    
            for t in alltime: #t=t+tw
                ep = int(t) # epoch to which retrieve the RBM

                if not(('W'+str(ep)) in f): # check for last time
                    break
        
                Wt = torch.tensor(f['W'+str(ep)]) 
                Wt = np.array(Wt) #W(t+tw)
                
                if perm:
                	W = self.PermMatrix(W_tw,Wt) 
                else:
                	W = Wt
 
                cc = (W-W_tw)**2

                alls.append(np.sum(cc)/(self.Nh*self.Nv))
                allt.append(ep-ttw)
        
            allt = np.array(allt)

            plt.loglog(np.array(allt),alls,label='$t_w$ = {}'.format(ttw))
    
        plt.legend()
        plt.title(' Two time weight correlation  $\longrightarrow$epochs = {}, $\eta$ = {}, $N_h$ = {}, gibbs steps = {}, T = {},  RDM = {}, Reg = {} '.format(self.emax,self.lr, self.Nh, self.gibbs_steps, self.T, self.rdm, self.regu), fontsize = 6)
        plt.xlabel('t')
        plt.ylabel('$\Delta$(t+$t_w$,$t_w$)')
        plt.show()
        
    def PlotMSDbias(self,bias):
    
        plt.figure(dpi=150)
        f = h5py.File(self.fname,'r')
        
        if bias == 'h':
            name = 'bias_h'
        else:
            name = 'bias_v'
            
        timew = []
        for n in range (1,30):
            tw = 2**(2*n) #para coger un medio de los tw
            timew.append(tw)   
             
        timew = np.array(list(set(timew)))
        timew = np.sort(timew)  

        for ttw in timew:
            alls = []
            epw = int(ttw)
    
            if not((name+str(epw)) in f): # check for last time
                break    
            
            b_tw = torch.tensor(f[name+str(epw)]) 
            b_tw = np.array(b_tw) #b(tw)
    
            alltime = []
            allt = []
            
            for m in range (1,30):
                tt = ttw + 2**m
                alltime.append(tt) 
                
            alltime = np.array(list(set(alltime)))
            alltime = np.sort(alltime)
    
            for t in alltime: #t=t+tw
                ep = int(t) # epoch to which retrieve the RBM

                if not((name+str(ep)) in f): # check for last time
                    break
        
                b = torch.tensor(f[name+str(ep)]) 
                b = np.array(b) #b(t+tw)
 
                cc = (b-b_tw)**2

                alls.append(np.mean(cc))
                allt.append(ep-ttw)
        
            allt = np.array(allt)

            plt.loglog(np.array(allt),alls,label='$t_w$ = {}'.format(ttw))
    
        plt.legend()
        plt.title(' Two time {} correlation  $\longrightarrow$epochs = {}, $\eta$ = {}, $N_h$ = {}, gibbs steps = {}, T = {}, RDM = {}, Reg = {} '.format(name, self.emax,self.lr, self.Nh, self.gibbs_steps, self.T, self.rdm, self.regu), fontsize = 6)
        plt.xlabel('t')
        plt.ylabel('$\Delta$(t+$t_w$,$t_w$)')
        plt.show()
        
   
    def PlotSusceptibilidadW(self):
    
        plt.figure(dpi=300)
        f = h5py.File(self.fname,'r')

        timew = []
        lim = []
        
        for n in range (1,60):
        	tw = 2**(2*n) #para coger un medio de los tw
        	timew.append(tw)
        	
        timew = np.array(list(set(timew)))
        timew = np.sort(timew)
        
        for ttw in timew:
        	alls = []
        	epw = int(ttw)
        	
        	if not(('W'+str(epw)) in f): # check for last time
        		break
        	
        	W_tw = torch.tensor(f['W'+str(epw)])
        	W_tw = np.array(W_tw) #W(tw)
        	
        	alltime = []
        	allt = []
        	
        	for m in range (1,30):
        		tt = ttw + 2**m
        		alltime.append(tt)
        		
        	alltime = np.array(list(set(alltime)))
        	alltime = np.sort(alltime)
        	
        	for t in alltime: #t=t+tw
        		ep = int(t) # epoch to which retrieve the RBM
        		
        		if not(('W'+str(ep)) in f): # check for last time
        			break
        			
        			
        		W = torch.tensor(f['W'+str(ep)])
        		W = np.array(W) #W(t+tw)
        		
        		cc = (W-W_tw)**2
        		
        		alls.append(np.mean(cc))
        		allt.append(ep-ttw)
        	
        	allt = np.array(allt)
        	lim.append(np.max(alls))
        	
        	plt.loglog(np.array(allt),alls,label='$t_w$ = {}'.format(ttw))
        	
        plt.legend()
        plt.title(' Two time weight correlation  $\longrightarrow$epochs = {}, $\eta$ = {}, $N_h$ = {}, gibbs steps = {}, T = {},  RDM = {}, Reg = {} '.format(self.emax,self.lr, self.Nh, self.gibbs_steps,self.T, self.rdm, self.regu), fontsize = 6)
        plt.xlabel('t')
        plt.ylabel('$\Delta$(t+$t_w$,$t_w$)')
        plt.show()
        
        U = np.max(lim)
        
        plt.figure(dpi=300)
        for ttw in timew:
        	alls = []
        	epw = int(ttw)
        	
        	if not(('W'+str(epw)) in f): # check for last time
        		break
        		
        	W_tw = torch.tensor(f['W'+str(epw)])
        	W_tw = np.array(W_tw) #W(tw)
        	
        	alltime = []
        	allt = []
        	
        	for m in range (1,30):
        		tt = ttw + 2**m
        		alltime.append(tt)
        		
        	alltime = np.array(list(set(alltime)))
        	alltime = np.sort(alltime)
        	
        	for t in alltime: #t=t+tw
        		ep = int(t) # epoch to which retrieve the RBM
        		
        		if not(('W'+str(ep)) in f): # check for last time
        			break
        			
        		W = torch.tensor(f['W'+str(ep)])
        		W = np.array(W) #W(t+tw)
        		
        		cc = (W-W_tw)**2
        		
        		chi = np.abs(cc - np.full_like(W, U))
        		
        		alls.append(np.mean(chi))
        		allt.append(ep-ttw)
        		
        	allt = np.array(allt)
        	
        	plt.loglog(allt,alls,label='$t_w$ = {}'.format(ttw))
        	
        plt.legend()
        plt.title(' $\chi$  $\longrightarrow$epochs = {}, $\eta$ = {}, $N_h$ = {}, gibbs steps = {}, T = {},  RDM = {}, Reg = {} '.format(self.emax,self.lr, self.Nh, self.gibbs_steps, self.T, self.rdm, self.regu), fontsize = 6)
        plt.xlabel('t')
        plt.ylabel('$\chi$')
        plt.show()
        
    def Wimshow(self,columnas):
        
        plt.figure(dpi=150)
        f = h5py.File(self.fname,'r')
        	
        time = [1]
        for n in range(1,10):
        	for m in range(1,10):
        		t = 2**(3*n) + 2**(3*m)
        		time.append(t)
        			
        time = np.array(list(set(time)))
        time = np.sort(time)
        allt = []
        	
        W = []
        	
        
        for t in time:
        	ep = int(t) # epoch to which retrieve the RBM
        	if not(('W'+str(ep)) in f): # check for last time
        		break
        	for col in columnas:	
        		W_t = np.array(f['W'+str(t)])[col,:]
        		W.append(W_t)
        		allt.append(t)
        			
        nr = int(np.shape(W)[0]/len(columnas)) #nrow - nro tiempos válidos
        r = self.ImConcat(torch.tensor(W),ncol=len(columnas),nrow=nr)
        plt.imshow(r,cmap='gray')
        return allt,nr
        	
        	
    def PermMatrix(self,W1,W2):
        
        C_ab = np.sum((W2[:, np.newaxis, :] - W1[np.newaxis, :, :]) ** 2, axis=2)
        row,col = linear_sum_assignment(C_ab)
        	
        M = W1[col,:] #ordeno 1 para parecerse a 2
        return M
        
    def resetfeaures(self):
    	f = h5py.File(self.fname,'r')
    	with h5py.File(self.fname, 'r') as archivo:
    		claves = list(archivo.keys())
    		if claves:
        		last_name = claves[-1]
        		num = re.findall(r'\d+', last_name)
        		tlast = int(num[0])
        		print(tlast)
       		
       			self.W = torch.tensor(f['W'+str(tlast)])
       			self.b_v = torch.tensor(f['bias_v'+str(tlast)])
       			self.b_h = torch.tensor(f['bias_h'+str(tlast)])  

    		else:
        		print("El archivo HDF5 está vacío.")
    	
            	
