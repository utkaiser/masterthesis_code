import numpy as np
from skimage.transform import resize
import generate_data.wave2 as wave2
import generate_data.ParallelCompute as PComp
import generate_data.WavePostprocess as WavePostprocess
import generate_data.WaveUtil as WaveUtil
import generate_data.OPPmodel as OPPmodel
#import modeltraining
#import CleanUtil

#import torch
#import torchvision
#import torchvision.transforms as transforms
#import torch.nn as nn


def InitNetParareal(u0,ut0,vel,dx,dt,cT,m,tm,T,pimax,net=None):
    '''
    Initialize parareal iteration
    compute initial guess solution from coarse solver
    with an option further process with JNet
    Input
    u0,ut0: initial data on fine grid
    vel: wavespeed on fine grid
    dx,dt: fine step size 
    cT: propagation time \delta t*
    m,tm: spatial, temporal step size ratio
    T: simulation time
    pimax: maximum iteration
    net: JNet
    Output
    up,utp: arrays of wavefields
    velX: wavespeed on coarse grid
    '''
    # Number of time slices - fine and coarse propagators communicate
    ncT = round(T/cT)
    Ny,Nx = vel.shape
    nx = round(Nx/m)
    ny = round(Ny/m)
    
    velX = resize(vel,[ny,nx],order=4)
    dX = dx*m
    dT = dt*tm
    
    # Store solution at every time slice and parareal iteration
    up = np.zeros([Ny,Nx,ncT,pimax])
    utp = np.zeros([Ny,Nx,ncT,pimax])
    
    # Set initial condition
    for i in range(pimax):
        up[:,:,0,i] = u0
        utp[:,:,0,i] = ut0
    
    UX = resize(u0,[ny,nx],order=4)
    UtX = resize(ut0,[ny,nx],order=4)
    
    # Initialize iteration with coarse solution
    for j in range(ncT-1):        
        UX,UtX = wave2.velocity_verlet_time_integrator(UX,UtX,velX,dX,dT,cT)
        if net != None:
            vX,vtX = WavePostprocess.ApplyJNet2WaveSol(UX,UtX,vel,dx,net,m)
            up[:,:,j+1,0] = vX
            utp[:,:,j+1,0] = vtX
            UX = resize(vX,[ny,nx],order=4)
            UtX = resize(vtX,[ny,nx],order=4)
        else:
            up[:,:,j+1,0] = resize(UX,[Ny,Nx],order=4)
            utp[:,:,j+1,0] = resize(UtX,[Ny,Nx],order=4)
        
    return up,utp,velX


# Main function of Neural Net aided parareal
def parareal2_NNpostprocess(u0,ut0,vel,dx,dt,cT,m,tm,T,pimax,net):
    '''
    JNet parareal iteration
    Input
    u0,ut0: initial data on fine grid
    vel: wavespeed on fine grid
    dx,dt: fine step size 
    cT: propagation time \delta t*
    m,tm: spatial, temporal step size ratio
    T: simulation time
    pimax: maximum iteration
    net: JNet
    Output
    up,utp: arrays of wavefields
    velX: wavespeed on coarse grid
    '''
    # Number of time slices - fine and coarse propagators communicate
    ncT = round(T/cT)
    Ny,Nx = vel.shape
    nx = round(Nx/m)
    ny = round(Ny/m)
    
    dX = dx*m
    dT = dt*tm

    up,utp,velX = InitNetParareal(u0,ut0,vel,dx,dt,cT,m,tm,T,pimax,net)
    # Parareal iteration 
    for parI in range(pimax-1):
        #### SUBJECT TO CHANGE TO MULTIPROCESSING
        # Parallel solution
        vx = up[:,:,:,parI]
        vtx = utp[:,:,:,parI]
        print('iteration',parI+1)
        
        UcX,UtcX,UfX,UtfX = PComp.ParallelCompute(vx,vtx,vel,velX,dx,dX,dt,dT,cT)    
            
        # Serial update     
        for j in range(ncT-1):
            w0 = resize(up[:,:,j,parI+1],[ny,nx],order=4)
            wt0 = resize(utp[:,:,j,parI+1],[ny,nx],order=4)
            wX,wtX = wave2.velocity_verlet_time_integrator(w0,wt0,velX,dX,dT,cT)
            uX,utX = WavePostprocess.ApplyJNet2WaveSol(wX,wtX,\
                                                      vel,dx,net,m)            
            
            vX,vtX = WavePostprocess.ApplyJNet2WaveSol(UcX[:,:,j+1],UtcX[:,:,j+1],\
                                                      vel,dx,net,m)
                                   
            up[:,:,j+1,parI+1] = UfX[:,:,j+1] + uX - vX
            utp[:,:,j+1,parI+1] = UtfX[:,:,j+1] + utX - vtX
    
    
    return up,utp



# Main function of Original parareal
def parareal2_Original(u0,ut0,vel,dx,dt,cT,m,tm,T,pimax):
    '''
    Classical parareal iteration
    Input
    u0,ut0: initial data on fine grid
    vel: wavespeed on fine grid
    dx,dt: fine step size 
    cT: propagation time \delta t*
    m,tm: spatial, temporal step size ratio
    T: simulation time
    pimax: maximum iteration
    Output
    up,utp: arrays of wavefields
    velX: wavespeed on coarse grid
    '''
    # Number of time slices - fine and coarse propagators communicate
    ncT = round(T/cT)
    Ny,Nx = vel.shape
    nx = round(Nx/m)
    ny = round(Ny/m)
    
    dX = dx*m
    dT = dt*tm
    
    up,utp,velX = InitNetParareal(u0,ut0,vel,dx,dt,cT,m,tm,T,pimax)
    
    # Parareal iteration 
    for parI in range(pimax-1):
        #### SUBJECT TO CHANGE TO MULTIPROCESSING
        # Parallel solution
        vx = up[:,:,:,parI]
        vtx = utp[:,:,:,parI]
        print('iteration',parI)
        
        UcX,UtcX,UfX,UtfX = PComp.ParallelCompute(vx,vtx,vel,velX,dx,dX,dt,dT,cT)          
        
        # Serial update     
        for j in range(ncT-1):
            w0 = resize(up[:,:,j,parI+1],[ny,nx],order=4)
            wt0 = resize(utp[:,:,j,parI+1],[ny,nx],order=4)
            uX,utX = wave2.velocity_verlet_time_integrator(w0,wt0,velX,dX,dT,cT)
        
            vX = resize(UcX[:,:,j+1],[Ny,Nx],order=4)
            vtX = resize(UtcX[:,:,j+1],[Ny,Nx],order=4)
                       
            up[:,:,j+1,parI+1] = UfX[:,:,j+1] + resize(uX,[Ny,Nx],order=4) - vX
            utp[:,:,j+1,parI+1] = UtfX[:,:,j+1] + resize(utX,[Ny,Nx],order=4) - vtX
    
    
    return up,utp

# Main function of Procrustes parareal
def parareal2_Procrustes(u0,ut0,vel,dx,dt,cT,m,tm,T,pimax):
    '''
    Procrustes parareal iteration
    Input
    u0,ut0: initial data on fine grid
    vel: wavespeed on fine grid
    dx,dt: fine step size 
    cT: propagation time \delta t*
    m,tm: spatial, temporal step size ratio
    T: simulation time
    pimax: maximum iteration
    Output
    up,utp: arrays of wavefields
    velX: wavespeed on coarse grid
    '''
    # Number of time slices - fine and coarse propagators communicate
    ncT = round(T/cT)
    Ny,Nx = vel.shape
    nx = round(Nx/m)
    ny = round(Ny/m)
    
    dX = dx*m
    dT = dt*tm
    
    up,utp,velX = InitNetParareal(u0,ut0,vel,dx,dt,cT,m,tm,T,pimax)
    
    # Parareal iteration 
    for parI in range(pimax-1):
        #### SUBJECT TO CHANGE TO MULTIPROCESSING
        # Parallel solution
        vx = up[:,:,:,parI]
        vtx = utp[:,:,:,parI]
        print('iteration',parI)
        
        UcX,UtcX,UfX,UtfX = PComp.ParallelCompute(vx,vtx,vel,velX,dx,dX,dt,dT,cT)    
        UcX = resize(UcX,[Ny,Nx],order=4)
        UtcX = resize(UtcX,[Ny,Nx],order=4)
        UcXdx,UcXdy,UtcXdt = WaveUtil.WaveEnergyComponentField(UcX,UtcX,vel,dx)
        UfXdx,UfXdy,UtfXdt = WaveUtil.WaveEnergyComponentField(UfX,UtfX,vel,dx)
        
        if parI == 0:
            P,S,Q = OPPmodel.ProcrustesShiftMap((UcXdx,UcXdy,UtcXdt),(UfXdx,UfXdy,UtfXdt),datmode='numpy')
        else:
            P,S,Q = OPPmodel.ProcrustesShiftMap((UcXdx,UcXdy,UtcXdt),(UfXdx,UfXdy,UtfXdt),(P,S,Q),datmode='numpy')
        
        # Serial update     
        for j in range(ncT-1):
            w0 = resize(up[:,:,j,parI+1],[ny,nx],order=4)
            wt0 = resize(utp[:,:,j,parI+1],[ny,nx],order=4)
            wX,wtX = wave2.velocity_verlet_time_integrator(w0,wt0,velX,dX,dT,cT)
            
            uX,utX = WavePostprocess.ApplyOPP2WaveSol(resize(wX,vel.shape,order=4),resize(wtX,vel.shape,order=4),\
                                                      vel,dx,(P,S,Q))
                       
            vX,vtX = WavePostprocess.ApplyOPP2WaveSol(resize(UcX[:,:,j+1],vel.shape,order=4),\
                                                      resize(UtcX[:,:,j+1],vel.shape,order=4),vel,dx,(P,S,Q))
            
            up[:,:,j+1,parI+1] = UfX[:,:,j+1] + uX - vX
            utp[:,:,j+1,parI+1] = UtfX[:,:,j+1] + utX - vtX   
            
      
    return up,utp

# Main function of Procrustes parareal
def parareal2_coarseProcrustes(u0,ut0,vel,dx,dt,cT,m,tm,T,pimax):
    '''
    Procrustes parareal iteration, where correction operator is for coarse grid
    Input
    u0,ut0: initial data on fine grid
    vel: wavespeed on fine grid
    dx,dt: fine step size 
    cT: propagation time \delta t*
    m,tm: spatial, temporal step size ratio
    T: simulation time
    pimax: maximum iteration
    Output
    up,utp: arrays of wavefields
    velX: wavespeed on coarse grid
    '''
    # Number of time slices - fine and coarse propagators communicate
    ncT = round(T/cT)
    Ny,Nx = vel.shape
    nx = round(Nx/m)
    ny = round(Ny/m)
    
    dX = dx*m
    dT = dt*tm
    
    up,utp,velX = InitNetParareal(u0,ut0,vel,dx,dt,cT,m,tm,T,pimax)
    
    # Parareal iteration 
    for parI in range(pimax-1):
        #### SUBJECT TO CHANGE TO MULTIPROCESSING
        # Parallel solution
        vx = up[:,:,:,parI]
        vtx = utp[:,:,:,parI]
        print('iteration',parI)
        
        UcX,UtcX,UfX,UtfX = PComp.ParallelCompute(vx,vtx,vel,velX,dx,dX,dt,dT,cT)    
        
        UcXdx,UcXdy,UtcXdt = WaveUtil.WaveEnergyComponentField(resize(UcX,[ny,nx],order=4),resize(UtcX,[ny,nx],order=4),velX,dX)
        UfXdx,UfXdy,UtfXdt = WaveUtil.WaveEnergyComponentField(resize(UfX,[ny,nx],order=4),resize(UtfX,[ny,nx],order=4),velX,dX)
        
        if parI == 0:
            P,S,Q = OPPmodel.ProcrustesShiftMap((UcXdx,UcXdy,UtcXdt),(UfXdx,UfXdy,UtfXdt),datmode='numpy')
        else:
            P,S,Q = OPPmodel.ProcrustesShiftMap((UcXdx,UcXdy,UtcXdt),(UfXdx,UfXdy,UtfXdt),(P,S,Q),datmode='numpy')
        
        # Serial update     
        for j in range(ncT-1):
            w0 = resize(up[:,:,j,parI+1],[ny,nx],order=4)
            wt0 = resize(utp[:,:,j,parI+1],[ny,nx],order=4)
            wX,wtX = wave2.velocity_verlet_time_integrator(w0,wt0,velX,dX,dT,cT)
            
            uX,utX = WavePostprocess.ApplyOPP2WaveSol(wX,wtX,velX,dX,(P,S,Q))
                       
            vX,vtX = WavePostprocess.ApplyOPP2WaveSol(resize(UcX[:,:,j+1],[ny,nx],order=4),resize(UtcX[:,:,j+1],[ny,nx],order=4),\
                                                      velX,dX,(P,S,Q))
            
            up[:,:,j+1,parI+1] = UfX[:,:,j+1] + resize(uX - vX,vel.shape,order=4) 
            utp[:,:,j+1,parI+1] = UtfX[:,:,j+1] + resize(utX - vtX,vel.shape,order=4) 
            
      
    return up,utp

def parareal2_NNseq(u0,ut0,vel,dx,dt,cT,m,tm,Ts,Tf,pimax,net):
    '''
    Sequential JNet parareal iteration
    Input
    u0,ut0: initial data on fine grid
    vel: wavespeed on fine grid
    dx,dt: fine step size 
    cT: propagation time \delta t*
    m,tm: spatial, temporal step size ratio
    Ts: sequence slice time
    Tf: simulation time
    pimax: maximum iteration
    net: JNet
    Output
    up,utp: arrays of wavefields
    velX: wavespeed on coarse grid
    '''
    # Number of time slices - fine and coarse propagators communicate
    Ny,Nx = vel.shape
    nx = round(Nx/m)
    ny = round(Ny/m)
    
    dX = dx*m
    dT = dt*tm
    
    fcT = round(Tf/Ts)
    scT = round(Ts/cT)
    
    up,utp,velX = InitNetParareal(u0,ut0,vel,dx,dt,cT,m,tm,Tf+cT,pimax)
    endseq = 0
    
    for k in range(fcT):
        useq,utseq,_ = InitNetParareal(up[:,:,endseq,-1],utp[:,:,endseq,-1],\
                                    vel,dx,dt,cT,m,tm,Ts,pimax)
        # Parareal iteration 
        for parI in range(pimax-1):
            # Parallel solution
            vx = useq[:,:,:,parI]
            vtx = utseq[:,:,:,parI]
            print('iteration',parI)
            
            UcX,UtcX,UfX,UtfX = PComp.ParallelCompute(vx,vtx,vel,velX,dx,dX,dt,dT,cT)    
                
            # Serial update     
            for j in range(scT-1):
                w0 = resize(useq[:,:,j,parI+1],[ny,nx],order=4)
                wt0 = resize(utseq[:,:,j,parI+1],[ny,nx],order=4)
                wX,wtX = wave2.velocity_verlet_time_integrator(w0,wt0,velX,dX,dT,cT)
                uX,utX = WavePostprocess.ApplyNet2WaveSol7in(useq[:,:,j,parI+1],utseq[:,:,j,parI+1],\
                                                          resize(wX,vel.shape,order=4),resize(wtX,vel.shape,order=4),\
                                                          vel,dx,net)            
                
                vX,vtX = WavePostprocess.ApplyNet2WaveSol7in(vx[:,:,j],vtx[:,:,j],\
                                                          UcX[:,:,j+1],UtcX[:,:,j+1],\
                                                          vel,dx,net)
                                       
                useq[:,:,j+1,parI+1] = UfX[:,:,j+1] + uX - vX
                utseq[:,:,j+1,parI+1] = UtfX[:,:,j+1] + utX - vtX
        
        endseq = (k+1)*scT
        up[:,:,k*scT:endseq,:] = useq[:,:,:,:]
        utp[:,:,k*scT:endseq,:] = utseq[:,:,:,:]
        
        up[:,:,endseq,-1],utp[:,:,endseq,-1] = wave2.velocity_verlet_time_integrator(up[:,:,endseq-1,-1],utp[:,:,endseq-1,-1],\
                                              vel,dx,dt,cT)
        
        
    return up,utp