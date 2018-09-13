import numpy as np #array magic from numpy 
from scipy.stats import lognorm #import the log-normal distribution for choosing sphere sizes from it 
import copy #to make deep copies which do not change the original array they are copied from
import time # to time stuff 
import mayavi
from mayavi import mlab
import itertools
from copy import deepcopy 


class sphere: 
    """generate a sphere with attributes radius, position, state, tracer, etc."""
    def __init__(self, radius, tracer_index, dims, grid_size, delta, seq_add=False): 
        self.radius = radius #initialize with stated radius 
        nx,ny,nz = dims 
        x = np.random.uniform(0,nx*grid_size)
        y = np.random.uniform(0,ny*grid_size)
        if seq_add:
            z = (nz+1)*grid_size
        else:
            z = np.random.uniform(grid_size,nz*grid_size)
            
        self.position = np.array([x,y,z])
        self.tag = str(tracer_index) #this is the tag to keep track of particles as they move around from iteration to iteration 
        self.state = None #this is the state which characterizes the settling mechanism of a particle 
        self.surface = None #this is a boolean val stating whether par
        self.entrainable = False
        self.active_iterations = 0
        self.P_D = 0.5#1-np.exp(self.active_iterations/10) #10 is a model parameter.. mean active iterations


def initialize_spheres(radii_array,dims,grid_size,delta,seq_add=False):
    """fill a list with spheres having random positions, sequential tags, and radii chosen from radii array in a box of size dims""" 
    out = []
    for t,r in enumerate(radii_array): #tracer index and radius 
        out.append(sphere(r, t, dims, grid_size, delta, seq_add = seq_add))
    if seq_add: 
        return out
    else: 
        return sorted(out, key = lambda x: x.position[-1]-x.radius) #return them sorted by height lowest to highest 

#now initialize a NXxNYxNZ grid  BUT ACTUALLY YOU WANT AN NX+2 x NY+2 by NZ grid    
def initialize_grid(dims): #initialize a grid to be filled with registered particles
    """make an NX by NY by NZ grid. Note that metagrids need to include wrapped boundary elements, so they should be NX+2 by NY+2 by NZ.""" 
    g = np.empty(dims,dtype='object') #empty grid
    for s in itertools.product(*tuple(range(d) for d in dims)):
        g[s] = [] #need to be able to list append into the grid, as far as I know this is the only way
    return g #return a grid of empty lists 

def indices(p,grid_size): #return the indices with the particle p should occupy in the grid (or make them out of range if it doesn't fit!)
    """return the coordinates of a particle measured in units of grid_size""" 
    n = p.position/grid_size
    return np.array(np.floor(n).astype(int))

def mindices(p,grid_size): #the indices of a particle p in the mgrid with its periodic boundary conditions and two extra cells 
    """since the mgrid is NX+2 by NY+2, a given particle in the mgrid is indexed by its indices + [0,0,1]."""
    return indices(p,grid_size)+np.array([1,1,0])
    #keep in mind that particles reside in the core of mgrid-- the first and last rows and columns are redundant-
    
def itr(mgrid):
    """iterate through all particles within the mgrid, not including artificial boundary ones"""
    return [f for e in filter(lambda g: len(g)>0,mgrid[1:-1,1:-1,:].flatten()) for f in e]

def fullitr(mgrid): 
    return [f for e in filter(lambda g: len(g)>0,mgrid.flatten()) for f in e]
    
"""DEFINE FUNCTIONS TO REGISTER SPHERES INTO THE GRID"""    
def append_mgrid(s,grid_size,dims,mgrid): 
    """add a particle s to the mgrid, taking account of boundary addition as well.""" 
    NX,NY,NZ = dims #just unpacking for convenience 
    nx,ny,nz = mindices(s,grid_size)
    p = deepcopy(s)
    mgrid[nx,ny,nz].append(p) #so append the particle in its proper place 
    #then append the wrapped elements to their proper places. 
    if nx==1:#if the element is on the lower x boundary of the physical grid it needs to wrap past the upper and append
        p1 = deepcopy(s)
        p1.position += np.array([NX,0,0])*grid_size
        p1.tag = p1.tag+'w' #append tag to show this is a virtual element 
        mgrid[-1,ny,nz].append(p1)
    if nx==NX: #if the element is on the upper x boundary of the physical grid it needs to wrap past the lower and append
        p2 = deepcopy(s)
        p2.position += np.array([-NX,0,0])*grid_size
        p2.tag = p2.tag+'w'
        mgrid[0,ny,nz].append(p2)
    if ny==1:#if the element is on the lower y boundary of the physical grid it needs to wrap past upper y boundary 
        p3 = deepcopy(s)
        p3.position +=  np.array([0,NY,0])*grid_size
        p3.tag = p3.tag+'w'
        mgrid[nx,-1,nz].append(p3) #the element is on the lower corner
    if ny==NY: 
        p4 = deepcopy(s)
        p4.position += np.array([0,-NY,0])*grid_size
        p4.tag = p4.tag+'w'
        mgrid[nx,0,nz].append(p4) 
    if nx==1 and ny==1:  #now do the four corner wraps 
        p5 = deepcopy(s)
        p5.position += np.array([NX,NY,0])*grid_size
        p5.tag = p5.tag+'w'
        mgrid[-1,-1,nz].append(p5)
    if nx==NX and ny==1: 
        p6 = deepcopy(s)
        p6.position += np.array([-NX,NY,0])*grid_size
        p6.tag = p6.tag+'w'
        mgrid[0,-1,nz].append(p6)
    if nx==1 and ny==NY: 
        p7 = deepcopy(s)
        p7.position += np.array([NX,-NY,0])*grid_size
        p7.tag = p7.tag+'w'
        mgrid[-1,0,nz].append(p7)
    if nx==NX and ny==NY: 
        p8 = deepcopy(s)
        p8.position += np.array([-NX,-NY,0])*grid_size
        p8.tag = p8.tag+'w'
        mgrid[0,0,nz].append(p8)

        
def R(s,grid_size,dims,mgrid): #register a sphere
    """When a sphere is successfully rearranged, append it to the mgrid."""
    append_mgrid(s,grid_size,dims,mgrid) #append registered particle to grid
    return s

"""DEFINE THE WRAPPING FUNCTIONS TO ENFORCE PERIODIC BOUNDARY CONDITIONS"""
#wrap 
def W(s,grid_size,dims): #this is to be used to wrap back a particle whose coordinates have left the physical grid
    """wrap particles which have left the grid back onto it by taking each coordinate xi modulo NI*grid_size"""
    NX,NY,NZ = dims
    p = copy.deepcopy(s)
    x,y,z = p.position
    nx,ny,nz = mindices(p,grid_size)
    xshift = NX*grid_size #a shift the size of the physical grid 
    yshift = NY*grid_size
    if 1<=nx<=NX and 1<=ny<=NY: #if x and y both on physical grid 
        pass
        #print('0')
    elif nx==0 and 1<=ny<=NY: #if x outside of physical grid on negative side
        x=x+xshift #shift x to positive side
    elif nx==NX+1 and 1<=ny<=NY: #if x outside of physical grid on positive side 
        x=x-xshift #shift x to negative side
    elif 1<=nx<=NX and ny==0: 
        y=y+yshift
    elif 1<=nx<=NX and ny==NY+1:
        y=y-yshift
    elif nx==0 and ny==0: 
        x=x+xshift
        y=y+yshift
    elif nx==NX+1 and ny==NY+1:
        x=x-xshift 
        y=y-yshift
    p.state=0

    p.position = np.array([x,y,z])
    return p 



"""DEFINE THE PROXIMITY FUNCTIONS"""
def distance(p1,p2): #the distance between p1 p2 ---particles [r,x,y,z]
    return np.linalg.norm(p1.position-p2.position)

def on_bottom(p,delta): #return True if the particle is on bottom and False if not
    if p.radius < p.position[-1] < p.radius+delta: 
        return True
    else: 
        return False 

def neighbors(s,grid_size,mgrid): #it is important the sphere be in the physical core of the grid 
    """search the 27 mgrid cells surrounding a particle s for all of its neighboring particles. """
    nx,ny,nz = mindices(s,grid_size) #the coordinates of the particle in the grid
    NX,NY,NZ = mgrid.shape      
    if nx==0:
        xslc = slice(0,nx+2,1)
    else:
        xslc = slice(nx-1,nx+2,1)
    if ny==0:
        yslc = slice(0,ny+2,1)
    else:
        yslc = slice(ny-1,ny+2,1)
    if nz==0:
        zslc = slice(nz,nz+2,1)
    else:
        zslc = slice(nz-1,nz+2,1)
    out = [e for f in filter(lambda g: len(g)>0,mgrid[xslc,yslc,zslc].flatten()) for e in f if e.tag!=s.tag and e.tag!=s.tag+'w']
    return out

def overlaps(s,grid_size,delta,mgrid):
    """find the overlapping particles of a sphere s"""
    NX,NY,NZ = mgrid[1:-1,1:-1].shape
    r1 = s.radius
    out = []
    for ele in neighbors(s,grid_size,mgrid): #for every neighbor of s 
        r2 = ele.radius
        if distance(s,ele)<r1+r2-delta: #if the neighbor is  more than delta overlapping s
            out.append(ele)  
    return np.array(out)

def contacts(s,grid_size,delta,mgrid): 
    """find all contacting spheres of a sphere s"""
    r1 = s.radius
    out = []
    for ele in neighbors(s,grid_size,mgrid):
        r2 = ele.radius
        if r1+r2-delta<=distance(s,ele)<=r1+r2+delta:
            out.append(ele)
    return np.array(out)

def new_contacts(s1,s2,grid_size,delta,mgrid): #new contacts gained as the sphere moved from s1 to s2 
    """find contacts added as a sphere moves from s1 to s2"""
    old = contacts(s1,grid_size,delta,mgrid) #contacts at configuration s1
    oldtags = [o.tag for o in old] #all of the tracer indices of old contacts before movement 
    new = contacts(s2,grid_size,delta,mgrid) #contacts at configuration s2
    out = [n for n in new if n.tag not in oldtags] #maybe better way to do this
    return np.array(out) 

def within_grid(s,grid_size,dims): #returns True if s is inside or on the boundary of the grid, False if not 
    """is a particle within the grid? Or not.."""
    NX,NY,NZ = dims 
    nx,ny,nz = indices(s,grid_size)
    if 0<=nx<NX and 0<=ny<NY:
        return True 
    else: 
        return False 

def c_point(c,s): #return the contact point [x,y,z] between s and its contact c 
    """what is the contact point shared between a sphere s and its contact c?"""
    xc = c.position
    xs = s.position
    rs = s.radius 
    rc = c.radius
    #actually should use the center of the overlap gap, instead of favoring the s side: this makes the interactions assymetric 
    return (xs-xc)/np.linalg.norm(xs-xc)*rc + xc

def c_sort(s,grid_size,delta,mgrid): #return the contacts of s sorted in order of the height of their contact point with s 
    """sort contacts in order of the height of their contact point."""
    if len(contacts(s,grid_size,delta,mgrid))>0:
        pts = sorted(contacts(s,grid_size,delta,mgrid), key=lambda c: c_point(c,s)[-1]) #sorted by height of contact point
    else: 
        pts = []
    return np.array(pts)

def c_above(s,grid_size,delta,mgrid): #the contacts with contact point above s 
    """all of the contacts with contact piont above the center of sphere s"""
    if len(contacts(s,grid_size,delta,mgrid))>0: #this conditional may not be necessary. I don't know what happens when sorted intakes None
        out = [c for c in contacts(s,grid_size,delta,mgrid) if c_point(c,s)[-1]>s.position[-1]]
    else:
        out = []
    return np.array(out) #only those c which have contact point above s 

def c_below(s,grid_size,delta,mgrid): #the contacts with contact point below s 
    """all of the contacts with contact piont below the center of sphere s"""
    if len(contacts(s,grid_size,delta,mgrid))>=0:
        out = [c for c in contacts(s,grid_size,delta,mgrid) if c_point(c,s)[-1]<=s.position[-1]]
    else:
        out=[]
def D(p,grid_size,delta,mgrid): #the dropper--this function drops p directly downward 
    """drop a sphere directly downward """
    movedir = np.array([0,0,-1],dtype=float)#the particle will move straight down 
    o = copy.deepcopy(p)
    debug = []
    print(str(len(overlaps(o,grid_size,delta,mgrid)))+' before D')
    if on_bottom(o,delta) or len(contacts(o,grid_size,delta,mgrid))>=3: #either it's okay where it was, or it needs to move
        o.state=5 #particle is stable
    else: #ok so it needs to move
        do = copy.deepcopy(o)
        do.position = o.position + 0.99*delta*movedir
        N_o = len(overlaps(do,grid_size,delta,mgrid))#number of overlaps generated by the movement
        N_c = len(new_contacts(p,do,grid_size,delta,mgrid)) #number of new contacts generated by the movement
        if N_o!=0: #if overlaps are generated the particle does not move 
            N_co = len(contacts(p,grid_size,delta,mgrid))
            if N_co>=3: #if particle can't move and has enough contacts to be stable
                o.state=5
            elif N_co==2:
                o.state=4
            else:
                print(' this should never happen. gimbal? ' )
        else:
            while True: 
                do.position = o.position + 0.99*delta*movedir
                N_o = len(overlaps(do,grid_size,delta,mgrid))#number of overlaps generated by the movement
                N_c = len(new_contacts(p,do,grid_size,delta,mgrid)) #number of new contacts generated by the movement
            #if it can move:
                if N_o==0:
                    #ok so it can move
                    #either moving makes new contacts, or it doesn't 
                    if N_c==0 and not on_bottom(do,delta):
                        #moving does not make new contacts
                        o = do #so do the movement
                        continue #and continue 
                    elif N_c==0 and on_bottom(do,delta):
                        o = do #this would be stability 
                        o.state=5
                        break 
                    else:
                        N_cons = len(contacts(do,grid_size,delta,mgrid))
                        #moving makes new contacts(including possibly the ground)
                        if on_bottom(do,delta):
                            o = do
                            o.state=5
                            break
                        elif N_cons>=3:
                            o = do
                            o.state=5
                            break
                        elif N_cons==2:
                            o = do
                            o.state=4
                            break
                        elif N_cons==1:
                            o = do
                            o.state=3
                            #gimbal?
                            break
                        else:
                            print('wut')
      
    print(str(len(overlaps(o,grid_size,delta,mgrid)))+' after D')
    return o
                    
                
                
                
                
                

    
"""THE TRANSLATION FUNCTIONS"""
"""DEFINE THE TRANSLATION FUNCTIONS"""
def rT(q,grid_size,dims,delta): #random displacement -- tested and works 
    """translate the particle at most distance grid_size in a random direction on the x-y plane""" 
    NX,NY,NZ = dims
    rando_travel_distance = 0 #the running value of the distance the particle has traveled so far undergoing the 
    random_angle = 2*np.pi*np.random.random()
    d = np.array([np.cos(random_angle),np.sin(random_angle),0])*grid_size #random translation by distance grid_size 
    dhat = d/np.linalg.norm(d) #direction of translation 
    oq = copy.deepcopy(q)   #the particle which will undergo random translation
    while rando_travel_distance<grid_size: #while the particle has not yet traveled its full random displacement distance 
        oq.position = q.position + dhat*0.99*delta
        if within_grid(oq,grid_size,dims):
            q.position = oq.position #move it in increments of delta/2
            rando_travel_distance += 0.99*delta #update the rando travel distance completed
        else:
            oq = W(oq,grid_size,dims)  #or if necessary wrap it back to the other side 
            q.position=oq.position
    q.state=0 
    return q 

def Ott(p,q): #this is the movement direction of unreg p due needed to remove overlap with a registered particle q 
    """calculate the movement direction particle p needs to move to escape overlap with particle q""" 
    V = p.position-q.position
    R = q.radius+p.radius
    D = distance(p,q)
    return V*(R/D-1)  #this comes from simple geometry-- consider the vector movement required to remove the overlap of two spheres
    #notice Ott vanishes when R == D and grows in magnitude as D<R -> D<<R  
    
def Ot(p,grid_size,delta,mgrid): #the sum of the movement directions required by from all overlappers of p
    """sum the movements a particle p should make to escape overlap with each individual overlapper of it over all overlappers"""
    ov = overlaps(p,grid_size,delta,mgrid)
    vect = np.sum([Ott(p,o) for o in ov],0) #sum the movement direction due to one overlapper q over all overlappers 
    vect[-1] = np.absolute(vect[-1])
    return vect/np.linalg.norm(vect) #then use it to normalize the vector to give only the movement direction needed to remove all overlaps

"""THE FUNCTION TO ROLL ABOUT ONE POINT"""
def rot(the,A,P,o): #roll sphere o about axis A and pivot point P through angle the in radians
    """incrementally roll a sphere around axis A and pivot piont C through angle the"""
    #from https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxnbGVubm11cnJheXxneDoyMTJiZTZlNzVlMjFiZTFi
    u,v,w = A #gimbal lock occurs when A is 0
    x,y,z = np.array(o.position)
    a,b,c = P
    xp = (a*(v**2+w**2)-u*(b*v+c*w-u*x-v*y-w*z))*(1-np.cos(the))+x*np.cos(the)+(-c*v+b*w-w*y+v*z)*np.sin(the)
    yp = (b*(u**2+w**2) -v*(a*u+c*w-u*x-v*y-w*z))*(1-np.cos(the))+y*np.cos(the)+(c*u-a*w+w*x-u*z)*np.sin(the)
    zp = (c*(u**2+v**2)-w*(a*u+b*v-u*x-v*y-w*z))*(1-np.cos(the))+z*np.cos(the)+(-b*u+a*v-v*x+u*y)*np.sin(the)
    return np.array([xp,yp,zp]) 

def T(p,grid_size,dims,delta,mgrid): #the translator--this function moves p away from of overlapping
    """translate the particle p to escape overlaps""" 
    NX,NY,NZ = dims
    T_max = 2*grid_size #the maximum travel distance which is actually a model parameter 
    movedir = Ot(p,grid_size,delta,mgrid) #define movedir in direction of Ot
    qq = copy.deepcopy(p)
    oq = copy.deepcopy(p)
    #while particle still has overlaps and it hasn't reached max travel dist
    travel_distance = 0
    while len(overlaps(qq,grid_size,delta,mgrid))>0 and travel_distance<T_max:
        oq.position = qq.position +0.99*delta*movedir
        if within_grid(oq,grid_size,dims): #if it's on the grid, translate
            qq = oq
            travel_distance += 0.99*delta #update the travel distance
        else: 
            qq = W(oq,grid_size,dims)
    #if max travel distance exceeded
    if travel_distance>=T_max: #then particle needs to undergo a random displacement 
        qq = rT(p,grid_size,dims,delta) #then make it undergo a random translation 
    qq.state=0
    return qq

def find_Odir(p,cons,grid_size,dims,delta,mgrid):
    NX,NY,NZ = dims 
    zees = [] #z value after rotation
    thes = [] #rotation angle increments to sweep through delta
    axes = [] #rotation axes
    pivs = [] #pivot points 
    x,y,z = p.position #unpack particle position
    for c in cons:
        c1,c2,c3 = P = c.position #pivot point
        dthe = 0.99*delta/np.linalg.norm(p.position-c.position) #unsigned rotation increment
        #mean_cpoint = 
        #OK will need to change rotation axis to go away from all contacts
        A = np.array([y-c2,c1-x,0])/np.linalg.norm(np.array([y-c2,c1-x,0])) #rotation axis 
        for sign in [1,-1]:
            o = copy.deepcopy(p) #a copy of p to test the rotation 
            dthe = sign*dthe #signed rotation increment
            o.position = rot(dthe,A,P,o) #perform the test rotation
            if not within_grid(o,grid_size,dims): #if test rotation left boundary, wrap it back
                o = W(o,grid_size,dims)
            Z = o.position[-1] #what is the new value of Z induced by the test rotation?
            Z0 = p.position[-1]
            N_ov = len(overlaps(o,grid_size,delta,mgrid)) #how many overlaps were generated by the test rotation?
            if  Z<Z0 and N_ov==0:#it o is lower than p and has no overlaps
                zees.append(Z) #add potential movement parameters 
                thes.append(dthe)
                axes.append(A)
                pivs.append(P)
    print(str(zees)+': zees')
    print(str(thes)+': thes')
    print(str(axes)+ ': axes')
    print(str([p.position])+' p position')
    print(str([c.position for c in cons])+': contacts')
    #now among the possibilities find the optimal set of parameters 
    if len(zees)>0: #find largest decrease in z 
        min_z, idx = min((val, idx) for (idx, val) in enumerate(zees)) #smallest z coordinate after rotation
        A = axes[idx] #largest decrease in z through rot for particle p induced by these parameters
        P = pivs[idx]
        dthe = thes[idx]
        return (dthe,A,P) #return the tuple of optimal rotation parameters
    else:
        return None #return statement that O fails 
           
def find_OOdir(p,cons,grid_size,dims,delta,mgrid):
    NX,NY,NZ = dims 
    zees = [] #z value after rotation
    thes = [] #rotation angle increments to sweep through delta
    axes = [] #rotation axes
    pivs = [] #pivot points 
    x,y,z = p.position #unpack particle position        
    for c0,c1 in itertools.combinations(cons, 2): #possible pairs of two contacts 
        x0 = c0.position
        x1 = c1.position
        A = (x1-x0)/np.linalg.norm(x1-x0)#rotation axis associated with c0,c1 
        P = np.dot(p.position-x0,A)*A+x0 #pivot point associated with c0,c1 
        dthe = 0.99*delta/np.linalg.norm(p.position-P) #unsigned rotation increment        
        for sign in [1,-1]:
            o = copy.deepcopy(p) #a copy of p to test the rotation 
            dthe = sign*dthe #signed rotation increment 
            o.position = rot(dthe,A,P,o) #perform the test rotation
            if not within_grid(o,grid_size,dims): #if test rotation left boundary, wrap it back
                o = W(o,grid_size,dims)
            Z = o.position[-1] #what is the new value of Z induced by the test rotation?
            Z0 = p.position[-1]
            N_ov = len(overlaps(o,grid_size,delta,mgrid)) #how many overlaps were generated by the test rotation?
            if  Z<Z0 and N_ov==0:#it o is lower than p and has no overlaps
                zees.append(Z) #add potential movement parameters 
                thes.append(dthe)
                axes.append(A)
                pivs.append(P)
    #now among the possibilities find the optimal set of parameters 
    if len(zees)>0: #find largest decrease in z 
        min_z, idx = min((val, idx) for (idx, val) in enumerate(zees)) #smallest z coordinate after rotation
        A = axes[idx] #largest decrease in z through rot for particle p induced by these parameters
        P = pivs[idx]
        dthe = thes[idx]
        return (dthe,A,P)
    else:
        return None

def O(p,cons,grid_size,dims,delta,mgrid): #roll p around its lowest contact until it finds a new contact 
    """roll a sphere p around a single contact until it finds a new contact"""
    o = copy.deepcopy(p) #a copy of p- it will move. 
    print(str(len(overlaps(o,grid_size,delta,mgrid)))+' before O')    
    
    params = find_Odir(p,cons,grid_size,dims,delta,mgrid) #the roll parameters or None if O fails
    if params is None: #if there are no suitable roll directions for O 
        o.state=1 #drop instead
    else: #there is a roll direction
        dthe,A,P = params #unpack the roll parameters
        while True:
            do = copy.deepcopy(o) #make a copy to test the movement
            do.position = rot(dthe,A,P,do) #move the copied particle
            if not within_grid(do,grid_size,dims): #if test rotation left boundary, wrap it back
                do = W(do,grid_size,dims)
            N_ov = len(overlaps(do,grid_size,delta,mgrid))
            if N_ov==0:
                N_c = len(new_contacts(p,do,grid_size,delta,mgrid))
                if N_c==0 and not on_bottom(do,delta):
                    o = do # no overlaps, no new contacts, and not on bottom.. keep going
                    continue
                elif on_bottom(do,delta):
                    o = do #found the bottom without generating overlaps
                    o.state=5 #stable 
                    break
                elif N_c!=0: #particle has gained a new contact: 
                    o = do 
                    o.state=0 #governer needs to determine what to do now that there's a new contact 
                    break
                else:
                    print('what goes here?')
            else: #movement generates overlaps
                #can only try a drop
                N_cons = len(contacts(o,grid_size,delta,mgrid)) #number of contacts
                if N_cons>=3:
                    o.state=5
                    break
                else: 
                    o.state=1
                    print('attempting drop out of overlap generating O')
                    print('generated overlap is '+str([o.position for o in overlaps(do,grid_size,delta,mgrid).position]))
                    break
    print(str(len(overlaps(o,grid_size,delta,mgrid)))+' after O')
    return o 
                
def OO(p,cons,grid_size,dims,delta,mgrid): #roll p around its lowest contact until it finds a new contact 
    """roll a sphere p around a single contact until it finds a new contact"""
    o = copy.deepcopy(p) #a copy of p- it will move. 
    print(str(len(overlaps(o,grid_size,delta,mgrid)))+' before OO')
    
    params = find_OOdir(p,cons,grid_size,dims,delta,mgrid) #the roll parameters or None if O fails
    if params is None: #if there are no suitable roll directions for OO
        o.state = 3 #O instead
    else: #there is a roll direction
        dthe,A,P = params #unpack the roll parameters
        while True:
            do = copy.deepcopy(o) #make a copy to move
            do.position = rot(dthe,A,P,do) #move the copied particle
            if not within_grid(do,grid_size,dims): #if rotation left boundary, wrap it back
                do = W(do,grid_size,dims)
                
            N_ov = len(overlaps(do,grid_size,delta,mgrid))
            if N_ov==0:
                N_c = len(new_contacts(p,do,grid_size,delta,mgrid))
                if N_c==0 and not on_bottom(do,delta):
                    o = do # no overlaps, no new contacts, and not on bottom.. keep going
                    continue
                elif on_bottom(do,delta):
                    o = do #found the bottom without generating overlaps
                    o.state=5 #stable 
                    break
                else: #particle has gained a new contact: 
                    o = do 
                    o.state=0 #governer needs to determine what to do now that there's a new contact 
                    break
            else: #if overlaps were generated,
                o.state=0
                break
                
               
    print(str(len(overlaps(o,grid_size,delta,mgrid)))+' after OO')
                
    return o              
    
"""THE GOVERNER"""
def G_sa(s,cons,grid_size,dims,delta,mgrid):
    """sequential addition governer. Tell the rearranger what needs to be done to a particle p. Does it roll with O? Does it drop with D? Does it roll with OO? state=1: D; state=3: O; state=4: OO; state=5: R"""  
    p = copy.deepcopy(s) 
    if on_bottom(p,delta): #if the particle is on the bottom with no overlaps it is stable 
        p.state=5
    elif len(cons)==0: #if it has no contacts it should drop 
        p.state=1 #drop with D
    elif len(cons)==1:
        p.state=3  
    elif len(cons)>=2:
        p.state=4 #then it should OO
    return p


"""REDEFINE REARRANGE"""
def rearrange_sa(p,grid_size,dims,delta,mgrid,debug): #rearrange a sphere p
    """sequentially add a sphere p. It is important that p does not initially overlap with any spheres already in the mgrid."""
    NX,NY,NZ = dims
    s = copy.deepcopy(p)
    s.state=0    
    while True: #this loop just passes the particle through G,D,W,O,OO,R as necessary
        print(s.position,s.radius,s.state)

        #print(s.state)
        cons = c_sort(s,grid_size,delta,mgrid)
        if s.state==0:
            s = G_sa(s,cons,grid_size,dims,delta,mgrid)
            continue 
        elif s.state==1:
            s,debug = D(s,grid_size,delta,mgrid)
            continue
        elif s.state==2:
            s = W(s,grid_size,dims)
            continue
        elif s.state==3:
            s = O(s,cons,grid_size,dims,delta,mgrid)
            continue 
        elif s.state==4: 
            s = OO(s,cons,grid_size,dims,delta,mgrid)
            continue
        elif s.state==5:
            if indices(s,grid_size)[-1] < NZ: 
                s = R(s,grid_size,dims,mgrid)
            else: 
                print('Particle past max height')
                s = None 
            break
        


"""VISUALIZATION WITH MAYAVI"""

def view(mgrid,cmap = 'BuPu', color=None ,resolution=8):
    x,y,z = np.array([s.position for s in itr(mgrid)]).T
    r = np.array([s.radius for s in itr(mgrid)])
    spheres = (x,y,z,r)
    if color is not None:
        mlab.points3d(*spheres,scale_factor=2,color=color)
    else: 
        mlab.points3d(*spheres,scale_factor=2,colormap=cmap)

        
def G_cr(s,cons,grid_size,dims,delta,mgrid):
    """collective rearrangement governer. Given an initial collection of spatially overlapping particles, rearrange them one by one until all overlaps are removed 
    state=0 means evaluate with governer
    state=1 means DROP
    state=2 means WRAP
    state=3 means O
    state=4 means OO
    state=5 means REGISTER
    state=6 means T 
    """
    NX,NY,NZ = dims 
    p = copy.deepcopy(s)
    if within_grid(p,grid_size,dims):#if the particle is on the grid
        if len(overlaps(p,grid_size,delta,mgrid))>0:#if the particle has overlaps
            p.state=6 #it should translate via T    
        else: #if the particle has no overlaps
            if on_bottom(p,delta): #if the particle is on the bottom with no overlaps it is stable  
                p.state=5
            elif len(cons)==0: #if it has no contacts it should drop 
                p.state=1 #drop with D
            else: #if the particle has contacts
                if len(cons)==1: #if the particle has one contact
                    c = cons[0] #here is the one contact 
                    zmin = c.position[-1]-c.radius-p.radius
                    if zmin-delta <  p.position[-1] < zmin+delta: #if particle is at bottom of rotation path...
                        p.state=1 #then it should drop. 
                    else: 
                        p.state=3 #then it should O 
                elif len(cons)==2: #IF THE PARTICLE HAS TWO CONTACTS
                    c0,c1 = cons #here are the contacts 
                    x0 = c0.position
                    x1 = c1.position
                    A = (x1-x0)/np.linalg.norm(x1-x0)#rotation axis associated with c0,c1 
                    P = np.dot(p.position-x0,A)*A+x0 #pivot point associated with c0,c1 
                    L = np.linalg.norm(p.position-P)#lever arm associated with c0,c1
                    zmin = P[-1] - np.linalg.norm(P-p.position)*np.sin(np.arccos(A[-1])) #the minimum z coordinate along the rotation path determined by A,P 
                    if zmin-delta<p.position[-1]<zmin+delta: #if particle is at its lowest possible position around the pivot point 
                        zcmin = cons[0].position[-1]-cons[0].radius #lowest z on the contact 
                        zpmax = p.radius + p.position[-1] #highest z on the particle 
                        if zpmax>zcmin+delta:#if it can go lower by rolling on the lower contact
                            p.state=3 #then it should O on the lower contact
                        else: #else it should drop immediately from both contacts 
                            p.state=1
                    else: #the particle is not at the lowest position around the OO pivot 
                        p.state=4 #then it should OO around the pivot 

                elif len(cons)==3: #if it has three contacts
                    p.state=4
                else: #if it has more than 3 contacts
                    p.state=5 #then it is stable             
    elif not within_grid(p,grid_size,dims): #IF NOT ON GRID
        p.state=2 #wrap needs to occur

    return p     



def rearrange_cr(p,grid_size,dims,delta,mgrid):
    
    """collective rearrangement function. Take in a particle s and move it until it has no overlaps and is stable on all other particles within mgrid.
    state=0 means evaluate with governer
    state=1 means DROP
    state=2 means WRAP
    state=3 means O
    state=4 means OO
    state=5 means REGISTER
    state=6 means T     """
    NX,NY,NZ = dims
    s = copy.deepcopy(p)
    s.state=0
    iter_number = 0 
    while True: #this loop just passes the particle through G,D,W,O,OO,R as necessary
        iter_number+=1
        if iter_number>=50: 
            print('iteration limit') 
            break
        cons = c_sort(s,grid_size,delta,mgrid)
        if s.state==0:
            s=G_cr(s,cons,grid_size,dims,delta,mgrid)
            continue 
        elif s.state==1:
            s=D(s,grid_size,delta,mgrid)
            continue
        elif s.state==2:
            s = W(s,grid_size,dims)
            continue
        elif s.state==3: 
            s = O(s,cons,grid_size,dims,delta,mgrid)
        elif s.state==4: 
            s = OO(s,cons,grid_size,dims,delta,mgrid)
            continue
        elif s.state==5:
            if indices(s,grid_size)[-1]>=0 and indices(s,grid_size)[-1] < NZ: 
                s = R(s,grid_size,dims,mgrid)
            else: 
                print('Particle past max height')
            break
        elif s.state==6: 
            s = T(s,grid_size,dims,delta,mgrid)
            continue 
    return s 



