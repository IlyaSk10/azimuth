import sys;sys.path.insert(0,r'v:\py_gr')

#import grad_wave.geometry as geometry
#import grad_wave.QV as QV
import geometry
import QV
import numba as nb
import numpy as np;
#from utils import *
#from . import geometry,QV

#,fs=200,T=10,)

norm=np.linalg.norm
nrm=lambda x,o=None: norm(np.array(x).reshape(-1),ord=o);
mrm= lambda x: np.mean(np.abs(np.array(x).reshape(-1)))

pi=np.pi
pi2=2*pi;
jpi2=2j*pi;





#@nb.jit(nopython=True,parallel=True,nogil=True)  
@nb.jit(nopython=True,nogil=True,fastmath=True)  
def _dsum_z(xxze,itc,Gtm,xout):
    L=xout.shape[0]
    Nc=xxze.shape[1]
    for n in nb.prange(L):
        for c in range(Nc):
            #xout[n]+=xxze[n,c]*Gtm[n,c];
            #xout[n]+=np.sign(Gtm[n,c]);
            #sgn=1 if Gtm[n,c]>0.0 else -1;
            #xout[n]+=xxze[n,c]*sgn;
            if Gtm[n,c]>0.0:
                xout[n]+=xxze[n,c]
            else:
                xout[n]-=xxze[n,c]
            
        
    return xout


def diff_sum_z(xx,Gt,m6,tc,dt=1./200):        
    xxz=xx[:,range(0,xx.shape[1],3)];
    Gtz=Gt[:,2,:,:];    
    itc=np.array(tc/dt,dtype=int);    
    shape=list(xxz.shape);
    Nxx=shape[0]
    shape[0]=Nxx+max(itc)+1
    xxze=np.zeros(shape,dtype=xxz.dtype)
    xxze[0:Nxx]=xxz
    xs=np.zeros(Nxx,dtype=xxz.dtype)
    Gtztc=np.array([Gtz[k,:,itc[k]] for k in range(Gtz.shape[0])]) 
    Gtm=np.einsum('nm,cm->nc',m6,Gtztc);
    _t=tic()
    r=_dsum_z(xxze,itc,Gtm,xs);
    toc('_dsum_z',_t)
    return r
    #np.einsum("nf,f->nf",this._gfs,this.ff**nD)
    pass



def _irfttn(G):
    n=len(G.shape)-1;
    return np.fft.irfft(G,axis=n,norm='ortho');

def geomQV(geom,qv):
    return np.einsum("nkm,nf->nkmf",geom,qv);
    #return geometry.M2mm_reshape(np.einsum("nkij,nf->nkijf",geom,qv));
    

class response_t(object):
    def __init__(this,xx,fs=200,T=100):
        
        this.gparams=geometry.gsp6(xx);
        this.fs=fs;
        
        this.N=N=int(T*fs);
        this.tt,this.dt=np.linspace(0,T,N, endpoint=True, retstep=True,dtype='d');
        this.ff=ff=np.fft.rfftfreq(N,1/fs);
        this.jw=jpi2*ff
    
    def _respf(this,Q,v,nD,fp,a3,naive):            
    #def _respf(this,Q,v,nD=2,fp=1e7,a3=1e-5,naive=False):        
        _,_,r=this.gparams
        q,t=QV.expKz_tret(this.ff,r,Q,v,nD,fp,a3,naive);
        #q=np.ones_like(q);
        return (v**-3)*q,t;
        
        
    def __call__(this,Q,vp,vs,fp=1e7,a3=1e-5,naive=False):
        return this.make_green(Q,vp,vs,fp,a3,naive);
        
        
    #def make_green(this,Q,vp,vs,fp=1e7,a3=1e-5,naive=False):    
    def make_green(this,Q,vp,vs,fp=1e3,a3=1e-6,naive=False):        
        #this._gfp,this._gfs=Gfp,Gfs=[(v**-3)*this._respf(Q,v,0,fp,a3) for v in (vp,vs) ];  
        #t_ret(f,r,Q,vp,nD=2,fp=1e7,a3=1e-5,naive=False)
        #this._gfp,this._gfs=Gfp,Gfs=[this._respf(Q,v,0,fp,a3,naive) for v in (vp,vs) ];  
        this._gfp,this._tp=Gfp,tp=this._respf(Q,vp,0,fp,a3,naive);  
        this._gfs,this._ts=Gfs,ts=this._respf(Q,vs,0,fp,a3,naive);
        
        _,_,r=this.gparams
        this.tp=r/vp;
        this.ts=r/vs;
        
        def dup3(t):
            t=np.array(t)
            t3=np.zeros([3,len(t)],dtype=t.dtype);
            t3[:]=t;
            return t3.T.reshape(-1);
        
        this.tp3=dup3(this.tp)
        this.ts3=dup3(this.ts)
        
        
        #Gfp,Gfs=[(v**-0)*this._respf(Q,v,0,fp,a3) for v in (vp,vs) ];  
        
        #gss=(np.einsum("nkp,nq,n->nkpq",en,g,br)+np.einsum("nkp,nq,n->nkpq",en,g,br));    
        '''
        this.Gfp=Gfp=np.einsum("nkij,nf->nkijf",this.gparams[0],Gfp);
        this.Gfs=Gfs=np.einsum("nkij,nf->nkijf",this.gparams[1],Gfs);
        this.Gf =Gf= this.Gfp+this.Gfs
        
        this.GfpM,this.GfsM,this.GfM=[geometry.M2mm_reshape(G) for G in [Gfp,Gfs,Gf]]       
        '''
        return this
    
    def _frec_mask(this,f,frecBand=None):
        
        if not frecBand is None:
            ff=this.ff;
            lb=frecBand['fLeft'];
            rb=frecBand['fRight'];            
            mask=(ff>=lb)&(ff<rb);
            f=f[:,mask];
            return f,mask
        else:            
            return f,np.ones(f.shape[1],dtype=bool)
        
    def gfxx(this,gfx,nD=0,frecBand=None):
        #ied=1j;
        ied=jpi2;
        f=np.einsum("nf,f->nf",gfx,(ied*this.ff)**nD);    
        return this._frec_mask(f,frecBand);
        
    
    def gfp(this,nD=2,frecBand=None):
        
        # ied=-1j;
        # #ied=1
        # f=np.einsum("nf,f->nf",this._gfp,(ied*this.ff)**nD);
        
        return this.gfxx(this._gfp,nD,frecBand)[0];
    
    def gfs(this,nD=2,frecBand=None):
        
        # ied=-1j;        
        # #ied=1
        # f=np.einsum("nf,f->nf",this._gfs,(ied*this.ff)**nD)
        # return this._frec_mask(f,frecBand);
        return this.gfxx(this._gfs,nD,frecBand)[0];
    
    
    @property
    def r(this):
        return this.gparams[2]; 
    
    def gtp(this,nD=2):
        return _irfttn(this.gfp(nD));
    def gts(this,nD=2):
        return _irfttn(this.gfs(nD));
    
    def Gfp(this,nD=2,frecBand=None):
        return geomQV(this.gparams[0],this.gfp(nD,frecBand=frecBand));
        
    def Gfs(this,nD=2,frecBand=None):
        return geomQV(this.gparams[1],this.gfs(nD,frecBand=frecBand));    
    
    def Gtp(this,nD=2):
        return geomQV(this.gparams[0],this.gtp(nD));
        
    def Gts(this,nD=2):
        return geomQV(this.gparams[1],this.gts(nD)); 
    
    def diff_sum_p(this,xx,mm,nD=1):
        Gt=this.Gtp(nD);
        return diff_sum_z(xx,Gt,mm,this.tp,dt=1.0/this.fs)
            
    def diff_sum_s(this,xx,mm,nD=1):
        Gt=this.Gts(nD);
        return diff_sum_z(xx,Gt,mm,this.ts,dt=1.0/this.fs)        
        
    

def reshape_resp(Gb):
    return Gb.reshape(Gb.shape[0]*Gb.shape[1],Gb.shape[2],-1).transpose([2,0,1])    
    
def signal(G,mxx=0.,myy=0.,mzz=0.,mxy=0.,mxz=0.,myz=0.):      
    return np.einsum('nkm...,m->nk...',G,[mxx,myy,mzz,mxy,mxz,myz]);
    
    
    
    




if __name__=='__main__':
    
    from all import *
    
    x=np.random.rand(7)
    xx=[[1,0,0],[10,0,0],[100,00,00],[1000,000,000],[2000,000,000],[3000,000,000],[4000,000,000]]
    xx=[[1,0,0],[10,0,0],[100,100,00],[0,1000,1000],[0,2000,1000],[0,3000,100],[0,0,4000]]
    #xx=[[1,0,0]]
    #xx=[[0,0,10],[0,0,20],[0,0,30],[0,0,40]]
    rs=response_t(xx,T=7,fs=15000);
    rs=response_t(xx,T=7,fs=3000);
    rs(Q=25e0,vp=3000,vs=1000,naive=0);
    
    
    
    #geometry.M2mm_reshape(rs.gparams[1])
    
    '''
    sf=rs.smf(M=[1,1,1])    
    st=rs.smt(M=[1,1,1])    
    
    print(nrm(st[0]),nrm(st[1]),nrm(st[2]))
    
    Gf=rs.GfpM;
    
    print('GfM',nrm(Gf[0]),nrm(Gf[1]),nrm(Gf[2]))
    
    G=rs.GtM;
    
    print('GtM',nrm(G[0]),nrm(G[1]),nrm(G[2]))
    '''
    qs=rs.Gts();
    qp=rs.Gtp();
    nD=1;
    gfp=rs.gfp(nD);
    nD=0
    #ml=matlab('dr');
    ml=matlab();
    gtp0=rs.gtp(0);
    gtp1=rs.gtp(1);
    gtp2=rs.gtp(2);
    #ml.putvar('s',jsc(ff=rs.ff,tt=rs.tt,p0=gfp0,p1=gfp1,p2=gfp2));
    ml.putvar('t',rs.tt)
    ml.putvar('fs',rs.fs);
    ml.putvar('p0',gtp0);
    ml.putvar('p1',gtp1);
    ml.putvar('p2',gtp2);
    
    ml.putvar('o',jsc(ff=rs.ff,tt=rs.tt,gfp=rs.gfp(nD),gfs=rs.gfs(nD),gtp=rs.gtp(nD),gts=rs.gts(nD)))
    ml('n=7;t=o.tt;plot(t,o.gtp(:,n),t,o.gts(:,n));legend p s')
    #rs.gparams[1],'"',geometry.M2mm_reshape(rs.gparams[1])

