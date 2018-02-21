############################################################################
#for lie algebra computations in gl(n)
############################################################################

'''
gl takes a dictionary d and returns an element of the general linear algebra
with usual addition and scalar multiplication, and multiplication
(i.e. Lie bracket) is defined as the matrix commutator. The keys of d
must have form (i,j) where i,j are positive integers, and the value
of (i,j) is the scalar entry in position i,j of the matrix element.

Current supported scalar fields include only int, long, float, complex.

For example gl({(1,2):-5,(3,1):6}) corresponds to the matrix element
[[0,-5,0],[0,0,0],[6,0,0]].

Since multiplication is the Lie bracket, we have
gl({(1,1):1,(2,2):-1})*gl({(2,1):1})=gl({(2,1):-2})
'''

class gl:
    def __init__(self,d):
        self.d=d
        if any(type(x)!=tuple or len(x)!=2 for x in d.keys()):
            raise ValueError('invalid key')
        if any(type(x[0])!=int or type(x[1])!=int or x[0]<0 or x[1]<0 for x in d.keys()):
            raise ValueError('invalid key')
        if not all(isinstance(x, (int, long, float, complex)) for x in d.values()):
            raise ValueError('invalid key')

    def __str__(self):
        if len(self.d)==0:
            return '0'
        r=''
        def pretty(x):
            if x==1:
                return ''
            else:
                return x
        for k in self.d:
            if type(self.d[k])==complex:
                r+='+({})E({},{})'.format(self.d[k],k[0],k[1])
            elif self.d[k]>0:
                r+='+{}E({},{})'.format(pretty(self.d[k]),k[0],k[1])
            else:
                r+='-{}E({},{})'.format(pretty(-self.d[k]),k[0],k[1])
        if r[0]=='+':
            return r[1:]
        else:
            return r
        
    #vector addition
    def __add__(self,other):
        r={}
        keys=set(self.d)|set(other.d)
        for k in keys:
            value=self.d.get(k,0)+other.d.get(k,0)
            if value!=0:
                r[k]=value
        return gl(r)

    #vector subtraction
    def __sub__(self,other):
        r={}
        keys=set(self.d)|set(other.d)
        for k in keys:
            value=self.d.get(k,0)-other.d.get(k,0)
            if value!=0:
                r[k]=value
        return gl(r)

    #scalar multiplication
    def smult(self,c):
        for k in self.d:
            self.d[k]*=c

    #Lie bracket
    def __mul__(self,other):
        r={}
        for j in self.d:
            for k in other.d:
                if j[1]==k[0]:
                    r[(j[0],k[1])]=r.get((j[0],k[1]),0)+self.d[j]*other.d[k]
                if j[0]==k[1]:
                    r[(k[0],j[1])]=r.get((k[0],j[1]),0)-self.d[j]*other.d[k]
        return gl(r)
#end of class
    
############################################################################
#application: computing the index of a Lie algebra
############################################################################

import sympy

#creates an element corresponding to a matrix unit with 1 in position i,j
makegl=lambda i,j:gl({(i,j):1})

#index for the subalgebra defined by a list of basis vectors
def index(basis):
    m=len(basis)
    order=set()
    for b in basis:
        order=order|set(b.d.keys())
    order=list(order)
    n=len(order)
    x=list(sympy.symbols('x0:{}'.format(m)))
    A=sympy.Matrix(n, m, lambda i,j:basis[j].d.get(order[i],0))
    BM=sympy.zeros(m)
    for i in range(m):
        for j in range(i,m):
            v=basis[i]*basis[j]
            c=sympy.Matrix([v.d.get(k,0) for k in order])
            r=sympy.linsolve((A,c),x)
            if r==sympy.EmptySet():
                raise ValueError('given basis not closed under Lie bracket')
            r=sympy.linsolve((A,c),x).args[0]
            if len(r.free_symbols)>0:
                raise ValueError('given basis is linearly dependent')
            BM[i,j]=sum(map(lambda p:p[0]*p[1],zip(r,x)))
    BM-=BM.T
    #BM is the matrix of brackets of basis elements, where the ith basis
    #element is represented by the symbol xi
    #
    #sympy.pprint(BM)
    return len(BM.nullspace())
