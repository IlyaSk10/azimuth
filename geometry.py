import numpy as np

norm=np.linalg.norm
# 00 11 22 01 02 12     ; xx.shape=[N][3] -> [N][3,6]


nrm=lambda x: norm(np.array(x).reshape(-1));

def gamma_r(xx):
    xx=np.array(xx,dtype='d');    
    r=norm(xx,axis=1);
    br=1./r
    return np.einsum("n,ni->ni",br,xx),r,br

def M2mm(mxx=0,myy=0,mzz=0,mxy=0,mxz=0,myz=0):
    
    return np.array([[mxx,mxy,mxz],
                     [mxy,myy,myz],
                     [mxz,myz,mzz]       
                     ],dtype='d')
def mm2M(mm):
    i=[0,4,8,1,2,5]
    return np.array(mm,copy=False).reshape(-1)[i]

def Mfrob(mxx=0,myy=0,mzz=0,mxy=0,mxz=0,myz=0):
    return np.linalg.norm(M2mm(*[mxx,myy,mzz,mxy,mxz,myz]),ord='fro');

def KaramoryMagnutudeFrob(mxx=0,myy=0,mzz=0,mxy=0,mxz=0,myz=0):
    mf=Mfrob(*[mxx,myy,mzz,mxy,mxz,myz]);
    return 2.0/3.0*np.log10(1e7*mf)-10.7;
    

def M2mm_reshape(G):
    
    Nc=G.shape[0];
    
    G9=G.reshape(Nc,3,9,-1) if len(G.shape)>4 else G.reshape(Nc,3,9);
    #i=[0,4,8,1,2,3]
    i=[0,4,8,1,2,5]
    G6=G9[:,:,i];
    #G6[:,:,3:]*=2;
    return G6;
    
def ggg(xx):
    g,r,br=gamma_r(xx);
    g3=np.einsum("ni,nj,nk,n->nijk",g,g,g,br);
    #g3=np.einsum("ni,nj,nk->nijk",g,g,g);
    return g3,g,r,br

def gg(xx):
    xxtxx=np.einsum("ni,nj->nij",xx,xx);
    return xxtxx.reshape(xxtxx.shape[0],-1);
def gg6(xx):
    t=gg(xx);
    

def gsp(xx):
    gp,g,r,br=ggg(xx);
    N,D,_,_=gp.shape;
    en=np.einsum("n,ij->nij",np.ones(N),np.eye(D));    
    gss=0.5*(np.einsum("nkp,nq,n->nkpq",en,g,br)+np.einsum("nkp,nq,n->nkqp",en,g,br));    
    gss=np.einsum("nkp,nq,n->nkpq",en,g,br)    
    gs=gss-gp;
    return gp,gs,r;

def gsp6(xx):    
    gp,gs,r=gsp(xx);
    return M2mm_reshape(gp),M2mm_reshape(gs),r;

def _response(gp,gs,mm):
     rp=np.einsum("nkpq,pq->nk",gp,mm);
     rs=np.einsum("nkpq,pq->nk",gs,mm);    
     return rp,rs;
                  
def response(xx,M,fnegZ=False):
    grp,grs,r=gsp(xx);
    mm=M2mm(*M);   
    if fnegZ:
        grp[:,2]=-grp[:,2]
        grs[:,2]=-grs[:,2]
        p=np.array([[1,0,0],[0,1,0],[0,0,-1]])
        mm=p@mm@p
    
    return _response(grp,grs,mm)+(r,);

    

if __name__=='__main__': 
    '''
    gs,gss,r=gsp([[1,0,0]])
    xx=[[1,2,3],[100,200,300],[0.100,0.200,0.300],[-1,2,-3]]
    #xx=[[1,0,0],[1,1,0],[1,1,1]]
    xn,r,br=gamma_r(xx)
    g3,g,r,br=ggg(xx)
    gs,gss,r=gsp(xx)
    rp,rs,r=response(xx,[1,1,1])
    '''
    rank=np.linalg.matrix_rank
    xx=[[1,2,3],[100,200,300],[0.100,0.200,0.300],[-1,2,-3]]
    xx=np.random.rand(13,3)
    xx=np.array(xx)
    xtx=gg(xx);
    g3,g,r,br=ggg(xx)
    sh=g3.shape
    a=g3.reshape(sh[0]*sh[1],sh[2]*sh[3])
    print('rank(a)=',rank(a))
    rk=np.linalg.matrix_rank(xtx)
    print('rank=',rk)
    a6,_,_=gsp6(xx)
    a6=a6.reshape(sh[0]*sh[1],-1)
    print('rank(a6)=',rank(a6))
    
    