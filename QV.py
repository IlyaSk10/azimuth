import numpy as np;
import numba as nb
pi=np.pi
pi2=2*pi;
pi4=4*pi;
eps=np.finfo(float).eps;

norm=np.linalg.norm
nrm=lambda x: norm(np.array(x).reshape(-1));
mrm= lambda x: np.mean(np.abs(np.array(x).reshape(-1)))

@nb.jit(nopython=True,nogil=True,fastmath=False)
def _tlog(t):
    if t>eps:
        return t*np.log(t);
    else:
        return 0.0;
        
tlog =np.vectorize(_tlog)    

def Kz(w,Q,vp,wp=2*pi*1e7,a3=1e-5):
    
    vr=vp*(1+np.log(wp)/(2*pi*Q));    
    #Kw=(1-wp/(pi*Q)*tlog(w/wp)/(1-(a3*w)**2))/vr;
    Kw=(w-wp/(pi*Q)*tlog(w/wp)/(1-(a3*w)**2))/vr;
    Gw=w/(2*vp*Q*(1+a3*w));
    return Kw,Gw;


def Kz_naive(w,Q,vp,wp=2*pi*1e7,a3=1e-5):    
    
    Kw=(w)/vp;
    Gw=w/(2*vp*Q);
    return Kw,Gw;

def t_ret(f,r,Q,vp,nD=2,fp=1e7,a3=1e-5,naive=False):
    w=pi2*np.array(f);
    wp=2*pi*fp;    
    Kzx=Kz_naive if  naive else Kz;    
    Kw,Gw=Kzx(w,Q,vp,wp,a3);    
    t=np.einsum("r,f->rf",r,Kw/w);
    return t
    
def expKz(f,r,Q,vp,nD=2,fp=1e7,a3=1e-5,naive=False):
    
    w=pi2*np.array(f);
    wp=2*pi*fp;
    jwnD=(1j*w)**nD;    
    Kzx=Kz_naive if  naive else Kz;    
    Kw,Gw=Kzx(w,Q,vp,wp,a3);    
    #K=1j*(Kw-0*w)-Gw;    
    K=-1j*Kw-Gw;    
    Kzr=np.einsum("r,f->rf",r,K);
    Grf=np.einsum("rf,f->rf",np.exp(Kzr),jwnD);
    return Grf/pi4;
    
    
def expKz_tret(f,r,Q,vp,nD=2,fp=1e7,a3=1e-5,naive=False):
    
    w=pi2*np.array(f);
    wp=2*pi*fp;
    jwnD=(1j*w)**nD;    
    Kzx=Kz_naive if  naive else Kz;    
    Kw,Gw=Kzx(w,Q,vp,wp,a3);    
    #K=1j*(Kw-0*w)-Gw;    
    K=-1j*Kw-Gw;    
    bvw=Kw/w;
    tret=np.einsum("r,f->rf",r,bvw);
    Kzr=np.einsum("r,f->rf",r,K);
    Grf=np.einsum("rf,f->rf",np.exp(Kzr),jwnD);
    return Grf/pi4,tret;

    

if __name__=='__main__':
    x=np.random.rand(10000)
    f=[0,1,2.1,4];
    
    G=expKz(f=f,vp=1000,r=[1,100,1000],Q=1);
    
    print(nrm(G[0]),nrm(G[1]),nrm(G[2]))