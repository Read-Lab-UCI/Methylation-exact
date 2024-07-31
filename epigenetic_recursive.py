import numpy as np
from scipy.sparse import diags,eye,bmat,coo_array
from scipy.sparse.linalg import eigs
import sparse
import matplotlib.pyplot as plt
from time import time
import torch

class recursive_collaroration:
    def __init__(self,nsites,param,collab=None,d=None,lamda=30,check_valid=False):
        #nsites are the number of CpG sites
        #param is the kinetic rate for standard model
        #collab is the kinetic rate for the collaboration force
        #if d in not none, it should be list/array, when d is shape 1, the CpGs are equally spaced
        #when shape==nsites, d is the cartesian position of CpG sites
        self.param=param
        self.nsites=nsites
        self.collab=collab
        self.d=np.array(d)
        self.lamda=lamda
        self.base=self.recursive_transition(nsites)
        if self.d is not None:
            if len(self.d)!=1 and len(self.d)!=self.nsites:
                print('wrong input for d(distance), d should be a scalar of a vector of lenght nsites,dictating the position of each sites')
                return
            self.distance=self.find_distance()
        if self.collab is not None:
            self.collab_mat = self.find_collab_recursive(self.nsites)['total']
            if check_valid:
                temp=self.collab_trasition()
                temp = coo_array((temp, (self.base.row, self.base.col)))
                print('two methods output the same matrix:{}'.format(np.isclose((temp - self.collab_mat).data, 10 ** -12).all()))
                #self.base.data=self.base.data+self.collab_mat
        else:
            self.collab_mat=0
        self.base = (self.base + self.collab_mat).tocoo()
        self.base.data=np.hstack((self.base.data,-np.array(self.base.sum(axis=0)).ravel()))
        self.base.row =np.hstack((self.base.row,np.arange(3**self.nsites)))
        self.base.col = np.hstack((self.base.col, np.arange(3**self.nsites)))
        #self.v=sparse.linalg.eigs(self.base, k=1, which='SM')

    def recursive_transition(self,nsites):
        #k_param[0]: u->h
        #k_param[1]: h->u
        #k_param[2]: h->m
        #k_param[3]: m->h
        if nsites==1:
            return diags([[self.param[0],self.param[2]],[self.param[1],self.param[3]]],offsets=[-1,1])
        blockdiag=self.recursive_transition(nsites-1)
        Identity=eye(blockdiag.shape[0])
        return bmat([[blockdiag,Identity*self.param[1],None],[Identity*self.param[0],blockdiag,Identity*self.param[-1]],[None,Identity*self.param[2],blockdiag]])

    def find_distance(self):
        distance=np.zeros((self.nsites,self.nsites))
        if len(self.d)==1:
            for i in range(1,self.nsites):
                distance=distance+np.diag([self.d[0]*i]*(self.nsites-i),k=i)+np.diag([self.d[0]*i]*(self.nsites-i),k=-i)
        else:
            for i in range(distance.shape[0]):
                distance[i,:]=np.abs(self.d-self.d[i])
        #return np.exp(-distance/self.lamda)
        return distance

    def collab_trasition(self):
        #h + h -> m + h
        #h + m -> m + m
        #u + h -> h + h
        #u + m -> h + m
        #h + u -> u + u
        #h + h -> u + h
        #m + u -> h + u
        #m + h -> h + h
        collab_mat=np.zeros(self.base.data.shape[0])
        for i in range(len(self.base.data)):
            index_curr=np.array(np.unravel_index(self.base.row[i],shape=[3]*self.nsites))
            index_from=np.array(np.unravel_index(self.base.col[i],shape=[3]*self.nsites))
            index_diff=np.argwhere(index_from!=index_curr).ravel()[0]
            if not index_from.any():
                continue
            if not (index_from-2).any():
                continue
            if index_from[index_diff]==0:
                if not np.delete(index_curr,index_diff).any():
                    continue
                effect=self.find_collab_effect(index_diff,index_from,[0,self.collab[2],self.collab[3]])
                collab_mat[i]=effect
                #self.base.data[i]=self.base.data[i]+effect
            elif index_from[index_diff]==2:
                if not (np.delete(index_curr,index_diff)-2).any():
                    continue
                effect = self.find_collab_effect(index_diff, index_from, [self.collab[-2], self.collab[-1],0])
                collab_mat[i]=effect
                #self.base.data[i] = self.base.data[i] + effect
            elif index_from[index_diff]-index_curr[index_diff]>0:
                if (np.delete(index_from,index_diff)-2).any():
                    effect = self.find_collab_effect(index_diff, index_from, [self.collab[4], self.collab[5],0])
                    collab_mat[i]=effect
                    #self.base.data[i] = self.base.data[i] + effect
            elif index_from[index_diff]-index_curr[index_diff]<0:
                if np.delete(index_from,index_diff).any():
                    effect = self.find_collab_effect(index_diff, index_from, [0, self.collab[0], self.collab[1]])
                    collab_mat[i]=effect
                    #self.base.data[i] = self.base.data[i] + effect
        #self.base.data=np.hstack((self.base.data,-np.array(self.base.sum(axis=0)).ravel()))
        #self.base.row =np.hstack((self.base.row,np.arange(self.base.shape[0])))
        #self.base.col = np.hstack((self.base.col, np.arange(self.base.shape[0])))
        collab_mat=np.array(collab_mat)
        return collab_mat

    def find_collab_effect(self,index_diff,index_curr,k):
        k=np.array(k)
        #effect=k[index_curr]*np.exp(-self.distance[index_diff]/self.lamda)
        effect=k[index_curr]*self.distance[index_diff]
        effect=np.delete(effect,index_diff)
        return np.sum(effect)

    def find_collab_recursive(self,larger_index):
        if larger_index==2:
            A=coo_array([[0,self.collab[4],0],[0,0,self.collab[6]],[0,0,0]])
            B=coo_array([[self.collab[4],0,0],[0,self.collab[5],0],[0,0,0]])
            C=coo_array([[0,0,0],[0,self.collab[2], 0], [0,0, self.collab[3]]])
            D=coo_array([[0,self.collab[5],0],[self.collab[2],0,self.collab[-1]],[0,self.collab[0],0]])
            E=coo_array([[self.collab[-2],0,0],[0,self.collab[-1],0],[0,0,0]])
            F=coo_array([[0,0,0],[0,self.collab[0],0],[0,0,self.collab[1]]])
            G=coo_array([[0,0,0],[self.collab[3],0,0],[0,self.collab[1],0]])
            return {1:{'A':A,'B':B,'C':C,'D':D,'E':E,'F':F,'G':G},'total':bmat([[A,B,None],[C,D,E],[None,F,G]])*self.distance[1][0]}
        last_dict=self.find_collab_recursive(larger_index-1)
        smaller_index=larger_index-1
        zeros=coo_array((3**(larger_index-2),3**(larger_index-2)))
        mat_dict=dict()
        for i in range(1,smaller_index):
            mat_dict[i]=dict()
            mat_dict[i]['B'] = bmat([[last_dict[i]['B'], None, None],[None, last_dict[i]['B'], None],[None, None, last_dict[i]['B']]])
            mat_dict[i]['E'] = bmat([[last_dict[i]['E'], None, None],[None, last_dict[i]['E'], None],[None, None, last_dict[i]['E']]])
            mat_dict[i]['C'] = bmat([[last_dict[i]['C'], None, None],[None, last_dict[i]['C'], None],[None, None, last_dict[i]['C']]])
            mat_dict[i]['F'] = bmat([[last_dict[i]['F'], None, None],[None, last_dict[i]['F'], None],[None, None, last_dict[i]['F']]])
            mat_dict[i]['A'] = bmat([[last_dict[i]['A'], None, None],[None, last_dict[i]['A'], None],[None, None, last_dict[i]['A']]])
            mat_dict[i]['D'] = bmat([[last_dict[i]['D'], None, None],[None, last_dict[i]['D'], None],[None, None, last_dict[i]['D']]])
            mat_dict[i]['G'] = bmat([[last_dict[i]['G'], None, None],[None, last_dict[i]['G'], None],[None, None, last_dict[i]['G']]])
        mat_dict[smaller_index]=dict()
        mat_dict[smaller_index]['B']=bmat([[diags([np.ones(3**(larger_index-2))*self.collab[4]],offsets=[0]),None,None],[None,diags([np.ones(3**(larger_index-2))*self.collab[5]],offsets=[0]),None],[None,None,zeros]])
        mat_dict[smaller_index]['E'] = bmat([[diags([np.ones(3 ** (larger_index - 2)) * self.collab[-2]], offsets=[0]), None, None],[None, diags([np.ones(3 ** (larger_index - 2)) * self.collab[-1]], offsets=[0]), None],[None, None, zeros]])
        mat_dict[smaller_index]['C']=bmat([[zeros,None,None],[None,diags([np.ones(3**(larger_index-2))*self.collab[2]],offsets=[0]),None],[None,None,diags([np.ones(3**(larger_index-2))*self.collab[3]],offsets=[0])]])
        mat_dict[smaller_index]['F']=bmat([[zeros,None,None],[None,diags([np.ones(3**(larger_index-2))*self.collab[0]],offsets=[0]),None],[None,None,diags([np.ones(3**(larger_index-2))*self.collab[1]],offsets=[0])]])
        mat_dict[smaller_index]['A'] = bmat([[None,diags([np.ones(3**(larger_index-2))*self.collab[4]],offsets=[0]), None], [None, None,diags([np.ones(3 ** (larger_index - 2)) * self.collab[6]], offsets=[0])], [zeros, None, None]])
        mat_dict[smaller_index]['D'] = bmat([[None,diags([np.ones(3**(larger_index-2))*self.collab[5]],offsets=[0]), None], [diags([np.ones(3**(larger_index-2))*self.collab[2]],offsets=[0]), None,diags([np.ones(3 ** (larger_index - 2)) * self.collab[-1]], offsets=[0])], [None, diags([np.ones(3**(larger_index-2))*self.collab[0]],offsets=[0]), None]])
        mat_dict[smaller_index]['G'] = bmat([[None, None,zeros],[diags([np.ones(3**(larger_index-2))*self.collab[3]],offsets=[0]), None,None], [None,diags([np.ones(3 ** (larger_index - 2)) * self.collab[1]], offsets=[0]),None]])
        total=coo_array((3**larger_index,3**larger_index))
        for i in range(1,larger_index):
            total=total+bmat([[mat_dict[i]['A'],mat_dict[i]['B'],None],[mat_dict[i]['C'],mat_dict[i]['D'],mat_dict[i]['E']],[None,mat_dict[i]['F'],mat_dict[i]['G']]])*self.distance[larger_index-1][i-1]
        #self.collab_mat_dict[larger_index]['diagonal_block']=sparse.bmat([[self.collab_mat_dict[smaller_index]['A'],None,None],[None,self.collab_mat_dict[1]['D'],None],[None,None,self.collab_mat_dict[1]['G']]])
        mat_dict['total']=total+bmat([[last_dict['total'],None,None],[None,last_dict['total'],None],[None,None,last_dict['total']]])
        return mat_dict

class recursive_collaroration_tensor:
    def __init__(self,nsites,param,collab=None,d=None,lamda=np.array([[30]])):
        #nsites are the number of CpG sites
        #param is the kinetic rate for standard model
        #collab is the kinetic rate for the collaboration force
        #if d in not none, it should be list/array, when d is shape 1, the CpGs are equally spaced
        #when shape==nsites, d is the cartesian position of CpG sites
        self.nsites=nsites
        if collab is not None:
            if d.shape[1]!=1 and d.shape[1]!=nsites:
                print('wrong input for d(distance), number of column of d should be 1 or the same as nsites')
                return
            self.param=np.repeat(param,repeats=collab.shape[0]*d.shape[0]*lamda.shape[0],axis=0)
            self.collab=np.tile(np.repeat(collab,repeats=d.shape[0]*lamda.shape[0],axis=0),reps=(param.shape[0],1))
            self.d=np.tile(np.repeat(d,repeats=lamda.shape[0],axis=0),reps=(param.shape[0]*collab.shape[0],1))
            self.lamda=np.tile(lamda,reps=(param.shape[0]*collab.shape[0]*d.shape[0],1))
            self.distance=self.find_distance()
        else:
            self.param=param
        self.base=self.recursive_transition(nsites)
        if self.collab is not None:
            self.collab_mat = self.find_collab_recursive(self.nsites)
            self.base = self.base + self.collab_mat
        else:
            self.collab_mat=0
        diagonal=self.base.sum(axis=1)
        diagonal=torch.sparse_coo_tensor(torch.vstack((diagonal.indices()[0],diagonal.indices()[1],diagonal.indices()[1])),diagonal.values(),size=self.base.shape)
        #diagonal=sparse.COO(np.vstack((diagonal.coords[0],diagonal.coords[1],diagonal.coords[1])),diagonal.data,shape=self.base.shape)

        self.base=self.base-diagonal

    def recursive_transition(self,nsites):
        k0=self.recursive_transition_ind_param(nsites,0)
        k1 = self.recursive_transition_ind_param(nsites, 1)
        k2 = self.recursive_transition_ind_param(nsites, 2)
        k3 = self.recursive_transition_ind_param(nsites, 3)
        k0=torch.sparse_coo_tensor(np.vstack((np.repeat(np.arange(self.param.shape[0]),repeats=k0.data.shape[0]),np.tile(k0.row,reps=(1,self.param.shape[0])),np.tile(k0.col,reps=(1,self.param.shape[0])))),(np.tile(k0.data,reps=(self.param.shape[0],1))*self.param[:,0][:,None]).ravel(),size=(self.param.shape[0],3**nsites,3**nsites))
        k1=torch.sparse_coo_tensor(np.vstack((np.repeat(np.arange(self.param.shape[0]),repeats=k1.data.shape[0]),np.tile(k1.row,reps=(1,self.param.shape[0])),np.tile(k1.col,reps=(1,self.param.shape[0])))),(np.tile(k1.data,reps=(self.param.shape[0],1))*self.param[:,1][:,None]).ravel(),size=(self.param.shape[0],3**nsites,3**nsites))
        k2=torch.sparse_coo_tensor(np.vstack((np.repeat(np.arange(self.param.shape[0]),repeats=k2.data.shape[0]),np.tile(k2.row,reps=(1,self.param.shape[0])),np.tile(k2.col,reps=(1,self.param.shape[0])))),(np.tile(k2.data,reps=(self.param.shape[0],1))*self.param[:,2][:,None]).ravel(),size=(self.param.shape[0],3**nsites,3**nsites))
        k3=torch.sparse_coo_tensor(np.vstack((np.repeat(np.arange(self.param.shape[0]),repeats=k3.data.shape[0]),np.tile(k3.row,reps=(1,self.param.shape[0])),np.tile(k3.col,reps=(1,self.param.shape[0])))),(np.tile(k3.data,reps=(self.param.shape[0],1))*self.param[:,3][:,None]).ravel(),size=(self.param.shape[0],3**nsites,3**nsites))
        #k0=sparse.COO(np.vstack((np.repeat(np.arange(self.param.shape[0]),repeats=k0.data.shape[0]),np.tile(k0.row,reps=(1,self.param.shape[0])),np.tile(k0.col,reps=(1,self.param.shape[0])))),(np.tile(k0.data,reps=(self.param.shape[0],1))*self.param[:,0][:,None]).ravel(),shape=(self.param.shape[0],3**nsites,3**nsites))
        #k1=sparse.COO(np.vstack((np.repeat(np.arange(self.param.shape[0]),repeats=k1.data.shape[0]),np.tile(k1.row,reps=(1,self.param.shape[0])),np.tile(k1.col,reps=(1,self.param.shape[0])))),(np.tile(k1.data,reps=(self.param.shape[0],1))*self.param[:,1][:,None]).ravel(),shape=(self.param.shape[0],3**nsites,3**nsites))
        #k2=sparse.COO(np.vstack((np.repeat(np.arange(self.param.shape[0]),repeats=k2.data.shape[0]),np.tile(k2.row,reps=(1,self.param.shape[0])),np.tile(k2.col,reps=(1,self.param.shape[0])))),(np.tile(k2.data,reps=(self.param.shape[0],1))*self.param[:,2][:,None]).ravel(),shape=(self.param.shape[0],3**nsites,3**nsites))
        #k3=sparse.COO(np.vstack((np.repeat(np.arange(self.param.shape[0]),repeats=k3.data.shape[0]),np.tile(k3.row,reps=(1,self.param.shape[0])),np.tile(k3.col,reps=(1,self.param.shape[0])))),(np.tile(k3.data,reps=(self.param.shape[0],1))*self.param[:,3][:,None]).ravel(),shape=(self.param.shape[0],3**nsites,3**nsites))
        k=k0+k1+k2+k3
        return k

    def recursive_transition_ind_param(self,nsites,index):
        #k_param[0]: u->h
        #k_param[1]: h->u
        #k_param[2]: h->m
        #k_param[3]: m->h
        if nsites==1:
            if index==0:
                k = coo_array([[0,0,0],[1,0,0],[0,0,0]])
            elif index==1:
                k = coo_array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
            elif index==2:
                k = coo_array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
            else:
                k= coo_array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
            return k
        blockdiag=self.recursive_transition_ind_param(nsites-1,index)
        Identity=eye(blockdiag.shape[0])
        if index==0:
            return bmat([[blockdiag,None,None],[Identity,blockdiag,None],[None,None,blockdiag]])
        elif index==1:
            return bmat([[blockdiag,Identity,None],[None,blockdiag,None],[None,None,blockdiag]])
        elif index==2:
            return bmat([[blockdiag,None,None],[None,blockdiag,None],[None,Identity,blockdiag]])
        else:
            return bmat([[blockdiag,None,None],[None,blockdiag,Identity],[None,None,blockdiag]])

    def find_distance(self):
        distance=np.zeros((self.param.shape[0],self.nsites,self.nsites))
        if self.d.shape[1]==1:
            d=np.arange(self.nsites)*self.d.ravel()[:,None]
        else:
            d=self.d
        for i in range(distance.shape[1]):
            distance[:,i,:]=np.abs(d-d[:,i][:,None])
        #return np.exp(-distance/self.lamda.squeeze()[:,None,None])
        return distance

    def find_collab_recursive_ind_param(self,larger_index,index):
        if larger_index==2:
            mat_dict=dict()
            mat_dict[1]=dict()
            zeros=coo_array((3,3))
            if index==0:
                mat_dict[1]['D']=coo_array([[0,0,0],[0,0,0],[0,1,0]])
                mat_dict[1]['F'] = coo_array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
                total=bmat([[zeros,None,None],[None,mat_dict[1]['D'],None],[None,mat_dict[1]['F'],zeros]])
            elif index==1:
                mat_dict[1]['F']=coo_array([[0,0,0],[0,0,0],[0,0,1]])
                mat_dict[1]['G']= coo_array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
                total=bmat([[zeros,None,None],[zeros,None,None],[None,mat_dict[1]['F'],mat_dict[1]['G']]])
            elif index==2:
                mat_dict[1]['C']=coo_array([[0,0,0],[0,1,0],[0,0,0]])
                mat_dict[1]['D'] = coo_array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
                total=bmat([[zeros,None,None],[mat_dict[1]['C'],mat_dict[1]['D'],None],[None,None,zeros]])
            elif index==3:
                mat_dict[1]['C']=coo_array([[0,0,0],[0,0,0],[0,0,1]])
                mat_dict[1]['G'] = coo_array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
                total=bmat([[None,zeros,None],[mat_dict[1]['C'],None,None],[None,None,mat_dict[1]['G']]])
            elif index==4:
                mat_dict[1]['A']=coo_array([[0,1,0],[0,0,0],[0,0,0]])
                mat_dict[1]['B'] = coo_array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
                total=bmat([[mat_dict[1]['A'],mat_dict[1]['B'],None],[None,None,zeros],[None,None,zeros]])
            elif index==5:
                mat_dict[1]['B']=coo_array([[0,0,0],[0,1,0],[0,0,0]])
                mat_dict[1]['D']= coo_array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
                total=bmat([[None,mat_dict[1]['B'],None],[None,mat_dict[1]['D'],zeros],[zeros,None,None]])
            elif index==6:
                mat_dict[1]['A']=coo_array([[0,0,0],[0,0,1],[0,0,0]])
                mat_dict[1]['E'] = coo_array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
                total=bmat([[mat_dict[1]['A'],None,None],[None,None,mat_dict[1]['E']],[None,zeros,None]])
            else:
                mat_dict[1]['E'] = coo_array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
                mat_dict[1]['D'] = coo_array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
                total=bmat([[zeros,None,None],[None,mat_dict[1]['D'],mat_dict[1]['E']],[zeros,None,None]])
            total=torch.sparse_coo_tensor(np.vstack((np.repeat(np.arange(self.param.shape[0]),repeats=total.data.shape[0]),np.tile(total.row,reps=(1,self.param.shape[0])),np.tile(total.col,reps=(1,self.param.shape[0])))).astype('int'),(np.tile(total.data,reps=(self.param.shape[0],1))*(self.collab[:,index]*self.distance[:,1,0])[:,None]).ravel(),size=(self.param.shape[0],9,9))
            mat_dict['total']=total
            return mat_dict
        last_dict=self.find_collab_recursive_ind_param(larger_index-1,index)
        smaller_index=larger_index-1
        zeros=coo_array((3**(larger_index-2),3**(larger_index-2)))
        mat_dict=dict()
        for i in range(1,smaller_index):
            mat_dict[i]=dict()
            for j in last_dict[i].keys():
                mat_dict[i][j] = bmat([[last_dict[i][j], None, None],[None, last_dict[i][j], None],[None, None, last_dict[i][j]]])
        mat_dict[smaller_index]=dict()
        identity = eye(3 ** (larger_index - 2))
        if index==0:
            mat_dict[smaller_index]['D'] = bmat([[zeros,None, None], [None, None,zeros], [None, identity, None]])
            mat_dict[smaller_index]['F']=bmat([[zeros,None,None],[None,identity,None],[None,None,zeros]])
        elif index==1:
            mat_dict[smaller_index]['F']=bmat([[zeros,None,None],[None,zeros,None],[None,None,identity]])
            mat_dict[smaller_index]['G'] = bmat([[None, None,zeros],[zeros, None,None], [None,identity,None]])
        elif index==2:
            mat_dict[smaller_index]['C']=bmat([[zeros,None,None],[None,identity,None],[None,None,zeros]])
            mat_dict[smaller_index]['D'] = bmat([[None,zeros, None], [identity, None,zeros], [None, zeros, None]])
        elif index==3:
            mat_dict[smaller_index]['C']=bmat([[zeros,None,None],[None,zeros,None],[None,None,identity]])
            mat_dict[smaller_index]['G'] = bmat([[None, None,zeros],[identity, None,None], [None,zeros,None]])
        elif index==4:
            mat_dict[smaller_index]['A'] = bmat([[None,identity, None], [None, None,zeros], [zeros, None, None]])
            mat_dict[smaller_index]['B']=bmat([[identity,None,None],[None,zeros,None],[None,None,zeros]])
        elif index==5:
            mat_dict[smaller_index]['B']=bmat([[zeros,None,None],[None,identity,None],[None,None,zeros]])
            mat_dict[smaller_index]['D'] = bmat([[None,identity, None], [zeros, None,zeros], [None, zeros, None]])
        elif index==6:
            mat_dict[smaller_index]['E'] = bmat([[identity, None, None],[None, zeros, None],[None, None, zeros]])
            mat_dict[smaller_index]['A'] = bmat([[None,zeros, None], [None, None,identity], [zeros, None, None]])
        else:
            mat_dict[smaller_index]['D'] = bmat([[None,zeros, None], [zeros, None,identity], [None, zeros, None]])
            mat_dict[smaller_index]['E'] = bmat([[zeros, None, None],[None, identity, None],[None, None, zeros]])
        total= torch.empty([self.param.shape[0], int(3**larger_index), int(3**larger_index)], layout=torch.sparse_coo)
        zeros=coo_array((3**(larger_index-1),3**(larger_index-1)))
        for i in range(1,larger_index):
            if index==0:
                temp=bmat([[zeros,zeros,zeros],[None,mat_dict[i]['D'],None],[None,mat_dict[i]['F'],None]])
            elif index==1:
                temp=bmat([[zeros,zeros,zeros],[zeros,zeros,zeros],[None,mat_dict[i]['F'],mat_dict[i]['G']]])
            elif index==2:
                temp=bmat([[zeros,zeros,zeros],[mat_dict[i]['C'],mat_dict[i]['D'],None],[zeros,zeros,zeros]])
            elif index==3:
                temp=bmat([[zeros,zeros,zeros],[mat_dict[i]['C'],None,None],[None,None,mat_dict[i]['G']]])
            elif index==4:
                temp=bmat([[mat_dict[i]['A'],mat_dict[i]['B'],None],[zeros,zeros,zeros],[zeros,zeros,zeros]])
            elif index==5:
                temp=bmat([[None,mat_dict[i]['B'],None],[None,mat_dict[i]['D'],None],[zeros,zeros,zeros]])
            elif index==6:
                temp=bmat([[mat_dict[i]['A'],None,None],[None,None,mat_dict[i]['E']],[zeros,zeros,zeros]])
            else:
                temp=bmat([[zeros,None,None],[None,mat_dict[i]['D'],mat_dict[i]['E']],[zeros,zeros,zeros]])
            temp=torch.sparse_coo_tensor(np.vstack((np.repeat(np.arange(self.param.shape[0]),repeats=temp.data.shape[0]),np.tile(temp.row,reps=(1,self.param.shape[0])),np.tile(temp.col,reps=(1,self.param.shape[0])))),(np.tile(temp.data,reps=(self.param.shape[0],1))*(self.collab[:,index]*self.distance[:,larger_index-1,i-1])[:,None]).ravel(),size=(self.param.shape[0],int(3**larger_index),int(3**larger_index)))
            total=total+temp
        temp=last_dict['total']
        temp=torch.sparse_coo_tensor(np.vstack((np.tile(temp._indices()[0],reps=3),(temp._indices()[1]+(torch.tensor([0,1,2])*3**(larger_index-1))[:,None]).ravel(),(temp._indices()[2]+(torch.tensor([0,1,2])*3**(larger_index-1))[:,None]).ravel())).astype('int'),np.tile(temp._values(),reps=3),size=(self.param.shape[0],int(3**larger_index),int(3**larger_index)))
        mat_dict['total']=total+temp
        return mat_dict

    def find_collab_recursive(self,larger_index):
        k0=self.find_collab_recursive_ind_param(larger_index,0)['total']
        k1=self.find_collab_recursive_ind_param(larger_index,1)['total']
        k2=self.find_collab_recursive_ind_param(larger_index,2)['total']
        k3=self.find_collab_recursive_ind_param(larger_index,3)['total']
        k4=self.find_collab_recursive_ind_param(larger_index,4)['total']
        k5=self.find_collab_recursive_ind_param(larger_index,5)['total']
        k6=self.find_collab_recursive_ind_param(larger_index,6)['total']
        k7=self.find_collab_recursive_ind_param(larger_index,7)['total']
        return k0+k1+k2+k3+k4+k5+k6+k7

if __name__=='__main__':
    #torch generation of batch of matrices is much faster
    #low time consumption to transform torch tensor to sparse tensor, 1/6 of tensor generation time
    #constant time 4~5s to transform sparse tensor to scipy sparse matrices in total
    nsites=8
    start=time()
    mat = recursive_collaroration_tensor(nsites, np.array([[1, 1, 1, 1],[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]]), np.array([[1, 2, 3, 4, 5, 6,7,8],[1, 2, 3, 4, 5, 6, 7, 8],[1, 2, 3, 4, 5, 6, 7, 8],[1, 2, 3, 4, 5, 6, 7, 8],[1, 2, 3, 4, 5, 6, 7, 8],[1, 2, 3, 4, 5, 6, 7, 8]]), d=np.array([[10],[5],[10],[5]]), lamda=np.array([[30],[60]]))
    print('time to construct matrix:{}'.format(time()-start))
    construct=0
    construct1=0
    solve=0
    solve1=0
    start=time()
    base=sparse.COO(mat.base._indices(),mat.base._values(),shape=(mat.param.shape[0],int(3**nsites),int(3**nsites)))
    print('time taken to change from torch tensor to sparse tensor:{}'.format(time()-start))
    for i in range(mat.param.shape[0]):
        start = time()
        temp=base[i].to_scipy_sparse()
        end=time()
        construct+=end-start
        E,V=eigs(temp,k=1,which='SM')
        solve+=time()-end
        start=time()
        mat1=recursive_collaroration(nsites,mat.param[i,:],mat.collab[i,:],d=mat.d[i,:],lamda=mat.lamda[i,:],check_valid=False)
        end=time()
        construct1+=end-start
        E1, V1 = eigs(mat1.base, k=1, which='SM')
        solve1+=time()-end
        if not np.isclose(np.abs(np.real(V))-np.abs(np.real(V1)),10**-11).all():
            print('not equal')
    print('time to re-construct matrix:{}'.format(construct))
    print('time to eigen decomposition of re-construct:{}'.format(solve))
    print('time to construct matrix by loop:{}'.format(construct1))
    print('time to eigen decomposition by loop:{}'.format(solve1))
    print('done')