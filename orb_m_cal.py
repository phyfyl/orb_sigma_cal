from matplotlib import pyplot
import numpy as np
import scipy.linalg as sla
import tinyarray
from math import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpi4py import MPI

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
print(rank,size)

tx = tinyarray.array([[0, 1], [1, 0]])
tz = tinyarray.array([[1, 0], [0, -1]])
ty = tinyarray.array([[0, -1j], [1j, 0]])
t0 = tinyarray.array([[1, 0], [0, 1]])
tp = (t0 + tz)/2
tm = (t0 - tz)/2
s0 = tinyarray.array([[1, 0], [0, 1]])
sx = tinyarray.array([[0, 1], [1, 0]])
sy = tinyarray.array([[0, -1j], [1j, 0]])
sz = tinyarray.array([[1, 0], [0, -1]])
sp = (sx + 1.0j*sy)/2
sm = (sx - 1.0j*sy)/2
kron = np.kron
sqrt=np.sqrt
sin=np.sin
cos=np.cos
exp=np.exp
pi=np.pi

echarge=1.602176634E-19
h=6.62607015E-34
ev2J=1.602E-19
mu_B = 9.274E-24
d=8.182E-9
a=4.334E-10
mu_0=1.25663706212E-6
g=2
Sc=a**2*sqrt(3)/2
ehbar=echarge/(2*pi*h)
Beff=mu_0*mu_B/(Sc*d)
 
def orbital_moment(us,ms,mu,kf,dk):
    # 参数设置
    Nk = int(2*kf/dk)      # k网格点数
    Nu = len(us)

    # 生成k网格（0到2π）
    kxs = np.linspace(-kf, kf, Nk)
    kys = np.linspace(-kf, kf, Nk)

    m_tri = np.zeros(Nu)
    m_topo = np.zeros(Nu)

    for iu in np.arange(rank,Nu,size):
        u=us[iu] 
        for ik in range(Nk**2):
            ikx=int((ik+0.1)/Nk)
            iky=ik-ikx*Nk
            kx = kxs[ikx]
            ky = kys[iky]
            # 计算哈密顿量的本征值和本征矢
            H_k = h_dirac(u,ms,kx, ky)
            E, U = np.linalg.eigh(H_k)

            # 计算速度算符在本征态基下的矩阵表示
            vx,vy = dh_dirac_dk(kx,ky)
            vx_eig = U.conj().T @ vx @ U  # 转换到本征基
            vy_eig = U.conj().T @ vy @ U 
        
            for n in range(2):
                for m in range(4):
                    if np.abs(E[m]-E[n])<0.000001:
                        continue
                    delta_E = E[m] - E[n]
                    Anm=(vx_eig[n,m]*vy_eig[m,n]).imag
                    m_tri[iu] += Anm/delta_E
                    m_topo[iu] += 2*(E[n]-mu)*Anm/delta_E**2 
    m_tri=ev2J*ehbar*(dk**2)*m_tri/mu_B*1E-18
    m_topo=ev2J*ehbar*(dk**2)*m_topo/mu_B*1E-18
    m_tri=comm.gather(m_tri,root=0)
    m_topo=comm.gather(m_topo,root=0)
    if rank==0:
        m_tri=np.sum(m_tri,axis=0)
        m_topo=np.sum(m_topo,axis=0)
        m_total=m_tri+m_topo 
        zeeman=g*mu_B*m_total*Sc*1E18*Beff/ev2J
        with open ('orbital_magnetization.dat','w') as f:
            f.write("electric_potential (eV)\tm_total (mu_B/nm^2)\tm_tri (mu_B/nm^2)\tm_topo (mu_B/nm^2)\t Zeeman energy (eV)\n")
            for i in range(Nu):
                f.write(f'{us[i]:.6e}\t{m_total[i]:.6e}\t{m_tri[i]:.6e}\t{m_topo[i]:.6e}\t{zeeman[i]:.6e}\n')

dk=0.01
kf=5
us=np.linspace(0,2,24)
mu=0
ms=np.array([[0,0,1],[0,0,-1]])
orbital_moment(us,ms,mu,kf,dk)
