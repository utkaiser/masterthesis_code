import numpy as np
import matplotlib.pyplot as plt
#import generate_data.WaveUtil as wutil

# Procrustean approach to shift coarse to fine solution

def ProcrustesShiftMap(it, coarse_dat,fine_dat,opmap=(np.empty(0),np.empty(0),np.empty(0)), vel = None, datmode='tensor'):
    # Compute Procustes shift map
    # a stable (hieu and richard paper)#########

    Ucx,Ucy,Utc = coarse_dat
    Ufx,Ufy,Utf = fine_dat
    U,S,V = opmap
    Cdat = serial_numpy_stack(Ucx,Ucy,Utc)
    Fdat = serial_numpy_stack(Ufx,Ufy,Utf)

    # f, ax = plt.subplots(2, 1)
    # f, ax = plt.subplots(2,10)
    # f.set_figheight(10)
    # f.set_figwidth(20)
    # f.suptitle(str(it) + " " + str(np.linalg.norm(Cdat-Fdat,ord='fro')))
    # dx = 2.0/128.0
    #print(str(it), (np.square(Ucx[:, :, 3] - Ucx[:,:,4])).mean(axis=None))
    # for i in range(10):
    #     uc, utc = wutil.WaveSol_from_EnergyComponent(Ucx[:,:,i], Ucy[:,:,i], Utc[:,:,i], vel, 4.0/128.0, 0)
    #     wc = wutil.WaveEnergyField(uc, utc, vel, 4.0/128.0)
    #     if i == 5:
    #         ax1 = f.add_subplot(2,1,1)
    #         pos1 = ax1.imshow(wc*dx*dx)
    #         plt.colorbar(pos1)
    #         ax2 = f.add_subplot(2, 1, 2)
    #         pos2 = ax2.imshow(tmp_wc*dx*dx)
    #         plt.colorbar(pos2)
    #         print(wc - tmp_wc)
    #     tmp_wc = wc
    #
    #     uf, utf = wutil.WaveSol_from_EnergyComponent(Ufx[:,:,i], Ufy[:,:,i], Utf[:,:,i], vel, 2.0 / 128.0, 0)
    #     wf = wutil.WaveEnergyField(uf, utf, vel, 2.0 / 128.0)

        # ax[0, i].imshow(wc)
        # ax[1, i].imshow(wf)


    #plt.show()

    return updateSVD(U,S,V,Fdat,Cdat)

def ProcrustesShift(U,V,coarse_arg,datmode='tensor'):
    Ux,Uy,Utc = coarse_arg
    
    if datmode == 'tensor':
        Cdat = serial_tensor_stack(Ux,Uy,Utc)
    elif datmode == 'numpy':
        Cdat = serial_numpy_stack(Ux,Uy,Utc)
    else:
        raise ValueError("datmode not defined")
    
    Fout0 = np.matmul(np.transpose(V),Cdat)
    Fout = np.matmul(U,Fout0)
    
    if datmode == 'tensor':
        uxout,uyout,utcout = deserial_tensor_stack(Fout,Ux.shape[0],Ux.shape[1],Ux.shape[2])    
    elif datmode == 'numpy':
        uxout,uyout,utcout = deserial_numpy_stack(Fout,Ux.shape[0],Ux.shape[1],Ux.shape[2])
    else:
        raise ValueError("datmode not defined")
    
    return uxout,uyout,utcout


# SVD of streaming data
def updateSVD(Uo,So,Vo,A,B):
    tol = 1e-15

    if (Uo.size == 0 or So.size==0 or Vo.size==0):
        QA,RA = np.linalg.qr(A,mode='reduced')
        QB,RB = np.linalg.qr(B,mode='reduced')
        
        up,sp,vtp = np.linalg.svd(np.matmul(RA,RB.transpose()))
        vp = vtp.transpose()
        ranknew = sum((sp/max(sp))>tol)
        
        Un = np.matmul(QA,up[:,:ranknew])
        Vn = np.matmul(QB,vp[:,:ranknew])
        Sn = sp[:ranknew]
        
        print('Coarse error:', np.linalg.norm(A-B,ord='fro'),
              '| OPP error:', np.linalg.norm(A-np.matmul(Un,np.matmul(Vn.transpose(),B)),ord='fro'))

    else:
        rankold = So.shape[0]
        Udim = Uo.shape[0]
        rownum = min(Udim,A.shape[1])
        
        UtA = np.matmul(Uo.transpose(),A)
        VtB = np.matmul(Vo.transpose(),B)
    
        QA,RA = np.linalg.qr(A - np.matmul(Uo,UtA),mode='reduced')
        QB,RB = np.linalg.qr(B - np.matmul(Vo,VtB),mode='reduced')
    
        term1 = np.matmul(np.concatenate((UtA,RA),axis=0),
                          np.concatenate((VtB,RB),axis=0).transpose())
        term2 = np.concatenate((np.concatenate((np.diag(So),np.zeros([rankold,rownum])),axis=1),
                               np.concatenate((np.zeros([rownum,rankold]),np.zeros([rownum,rownum])),axis=1)),
                                axis=0)
        K = term1 + term2 
        
        up,sp,vtp = np.linalg.svd(K)
        vp = vtp.transpose()
        
        ranknew = sum(sp/max(sp)>tol)
    
        Un = np.matmul(np.concatenate((Uo,QA),axis=1),up[:,:ranknew])
        Sn = sp[:ranknew]
        Vn = np.matmul(np.concatenate((Vo,QB),axis=1),vp[:,:ranknew])
        
        print('Coarse error:', np.linalg.norm(A-B,ord='fro'),
              '| OPP error:', np.linalg.norm(A-np.matmul(Un,np.matmul(Vn.transpose(),B)),ord='fro'))

    return Un,Sn,Vn

def serial_numpy_stack(ux,uy,utc):
    # Serialization and stacking of numpy format

    ny,nx,ns = ux.shape
    sux = np.reshape(ux,(ny*nx,ns))
    suy = np.reshape(uy,(ny*nx,ns))
    sutc = np.reshape(utc,(ny*nx,ns))
    udat = np.concatenate((sux,suy,sutc),axis=0)
    
    return udat

def deserial_numpy_stack(udat,ny,nx,ns):
    ux = np.reshape(udat[:ny*nx,:],(ny,nx,ns))
    uy = np.reshape(udat[ny*nx:2*ny*nx,:],(ny,nx,ns))
    utc = np.reshape(udat[2*ny*nx:,:],(ny,nx,ns))
    
    return ux,uy,utc

def serial_tensor_stack(ux,uy,utc):
    # Serialization and stacking of tensor format

    ns,ny,nx = ux.shape
    sux = np.transpose(np.reshape(ux,(ns,ny*nx)))
    suy = np.transpose(np.reshape(uy,(ns,ny*nx)))
    sutc = np.transpose(np.reshape(utc,(ns,ny*nx)))
    udat = np.concatenate((sux,suy,sutc),axis=0)
    
    return udat

def deserial_tensor_stack(udat,ns,ny,nx):

    ux = np.reshape(np.transpose(udat[:ny*nx,:]),(ns,ny,nx))
    uy = np.reshape(np.transpose(udat[ny*nx:2*ny*nx,:]),(ns,ny,nx))
    utc = np.reshape(np.transpose(udat[2*ny*nx:,:]),(ns,ny,nx))
    
    return ux,uy,utc