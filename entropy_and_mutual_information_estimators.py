import numpy as np
from itertools import combinations as icmb
from itertools import chain as ichain
import scipy.special as scis
import scipy.spatial as scispa

npn = np.newaxis
normVec = lambda x,r : (r[1]-r[0])*(x-x.min())/(x.max()-x.min()) + r[0]


def mylog(x):
    '''
    @details compute the logarithm avoiding singularities from the data
        log = mylog(x)

    @param x    (np.array) input vector
    @param log  (np.array) log of the input vector
    '''

    # get indices where x is valid (not 0, nan or inf). The last two may
    # arise if x is the result of a division and the denominator is 0
    ind = (x != 0) & (~np.isnan(x)) & (~np.isinf(x))

    logx = np.zeros(x.shape)   # init log as zeros

    #mylog[ind] = np.log(x[ind]) # compute the log only for valid x (base nat)
    logx[ind] = np.log2(x[ind]) # compute the log only for valid x (base 2)

    return logx

def pmf_single_var(data,nbins,lower_bound,upper_bound):
    '''
    @details compute the probability mass function of a single variable
        p = pmf_single_var(data,nbins,lower_bound,upper_bound)

    @param data          (np.array) data to compute the pmf
    @param nbins         (int) number of bins to use for the histogram
    @param lower_bound   (float) lower bound of the histogram
    @param upper_bound   (float) upper bound of the histogram

    @return p           (np.array) probability mass function
    '''

    hist,_ = np.histogram(data,bins=nbins,range=(lower_bound,upper_bound)) # unnormalized histogram
    return hist/hist.sum() # normalize the histogram to get the pmf

def entropy(p):
    '''
    @details compute the entropy of a discrete pdf
        H = entropy(p)

    @param p    (np.array) pdf of the signal
    @param H   (np.array) log of the input vector
    '''
    return -np.sum(p*mylog(p))


def entropy_nvars(p,ind):
    '''
    @brief compute the joint entropy given a pdf

    @details
        H = entropy_nvars(p,ind)

    intpus:
        @param p    N-dimensional np.array of the joint probability distribution
        @param ind  (tuple) list of indices over which the joint entropy is computed

    outputs:
        @param H   joint entropy

    Example: compute the joint entropy H(X0,X3,X7)
         H037 = entropy_nvars(p,(0,3,7))

    NOTE: entropy_nvars can be computed to compute the entropy for a single variable,
    but <ind> must be a tuple. Example:
        H1 = entropy_nvars(p,(1,))

    '''

    # compute the index of the variables not included in the entropy, to
    # sum over them:
    ind0 =tuple(set(range(len(p.shape))) - set(ind))

    return entropy(p.sum(axis=ind0))


def cond_entropy(p,ind1,ind2):
    '''
    @brief compute the conditional entropy given a pdf

    @details computhe the conditional entropy H(X1,...,|Xn,...,)
        H = cond_entropy(p,ind1,ind2)

    intpus:
        @param p    N-dimensional np.array of the joint probability distribution
        @param ind1  (tuple) list of indices over which the entropy is computed
        @param ind2  (tuple) list of indices of the conditioning variables

    outputs:
        @param H   conditional entropy

    Example: compute the joint entropy H(X0,X2|X7)
         H02c7 = cond_entropy(p,(0,2),(7,))

    NOTE: ind1 and ind2 must always be tuples even when a single variable is considered
    '''

    # H(X(v1)|X(v2)) =  H(X(v1),X(v2)) - H(X(v2))

    ## COMPUTE JOINT ENTROPY
    # set 1 and 2 are combined
    H12 = entropy_nvars(p,set(ind2)|set(ind1))

    ## COMPUTE H(X(v2)) ENTROPY
    H2 = entropy_nvars(p,set(ind2))

    return H12 - H2


def mutual_info(p,ind1,ind2):
    '''
    @brief compute the mutual info given a pdf

    @details computhe the mutual info between two set of variables I(X1,..., ; Y1,...,)
        I = mutual_info(p,ind1,ind2)

    intpus:
        @param p        N-dimensional np.array of the joint probability distribution
        @param ind1     (tuple) list of indices of the first set of variables
        @param ind2     (tuple) list of indices of the second set of variables

    outputs:
        @param I   mutual information

    Example: compute the mutual information I(X0,X5;X4,X2)
         I05a42 = cond_entropy(p,(0,5),(4,2))

    NOTE: ind1 and ind2 must always be tuples even when a single variable is considered
    '''

    # I(X(v1);X(v2)) = H(X(v1)) - H(X(v1)|X(v2))
    # NOTE: this is different from the co-information
    # I(X(v1);Y(v2)) = I(X(v1[0]), X(v1[1]), ... ; Y(v2[0]),...)
    return entropy_nvars(p,ind1) - cond_entropy(p,ind1,ind2)


def cond_mutual_info(p,ind1,ind2,ind3):
    '''
    @brief compute the conditional mutual info given a pdf

    @details computhe the mutual info between two set of variables conditioned to a third
    set, I(X1,..., ; Y1,..., | Z1,...)

        Ic = cond_mutual_info(p,ind1,ind2,ind3)

    intpus:
        @param p        N-dimensional np.array of the joint probability distribution
        @param ind1     (tuple) list of indices of the first set of variables
        @param ind2     (tuple) list of indices of the second set of variables
        @param ind3     (tuple) list of indices of the conditioning variables

    outputs:
        @param Ic      conditional mutual information

    Example: compute the conditional mutual information I(X0,X5;X4,X2|X1)
         I05a42c1 = cond_entropy(p,(0,5),(4,2),(1,))

    NOTE: ind1,ind2 and ind3 must always be tuples even when a single variable is
    considered
    '''

    # I(X(v1);X(v2)|X(v3)) = H(X(v1)|X(v3)) - H(X(v1)|X(v2),X(v3))
    ind4 = tuple(set(ind2) | set(ind3))
    return cond_entropy(p,ind1,ind3) - cond_entropy(p,ind1,ind4)


def KozLeoHestimator_nats(x, k : int = 1, norm = np.inf, mind = 0.):
    # Adapted from https://github.com/paulbrodersen/entropy_estimators/blob/master/entropy_estimators/continuous.py
    N, dim = x.shape 

    if norm == np.inf:
        log_cd = dim*np.log(2.)
    elif norm == 2:
        log_cd = (dim/2.)*np.log(np.pi) - np.log(scis.gamma(1 + dim/2.))
    else:
        raise NotImplementedError('norm has to be np.inf or 2')
    

    kdtree = scispa.KDTree(x)
    # query all points -- k+1 as query point also in initial set
    ei, _ = kdtree.query(x, k + 1, p = norm)
    dist = ei[:, -1]
    dist[dist < mind] = mind
    sumei = np.log(dist).sum()
    
    return scis.psi(N) - scis.psi(k) + log_cd + float(dim)/float(N)*sumei


def KraskovMI1_nats( x, y, k : int = 1 ):

    N, dim = x.shape 
  
    V = np.hstack([ x, y ])

    # Init query tree
    kdtree = scispa.KDTree( V )
    ei, _ = kdtree.query( V, k + 1, p = np.inf)
    # infty norm is gonna give us the maximum distance (x-dir or y-dir)
    dM = ei[:,-1] 
    
    kdtree_x = scispa.KDTree( x )
    kdtree_y = scispa.KDTree( y )

    nx = kdtree_x.query_ball_point( x, dM , p = np.inf, return_length = True)
    ny = kdtree_y.query_ball_point( y, dM , p = np.inf, return_length = True)

    # we do not add + 1 because it is accounted in query_ball_point
    ave = ( scis.psi( nx ) + scis.psi( ny ) ).mean()

    return scis.psi(k) - ave + scis.psi(N)


def cond_entropy_KozLeo_nats(x, y, **kozleoargs):
    '''
    @brief compute the conditional entropy (in nats)
        H( X | Y ) = H( X, Y ) - H( Y )
    Using the Kozalenko-Leonenko estimator

    @details The variables X and Y can have dx and dy
    dimensions, respectively. Namely:
        H( X1, X2 ... | Y1, Y2 ... )

    Usage:
        Hx_y = cond_entropy_KozLeo_nats( X, Y, k = 2)
    
    Inputs
        @param x        [nd.array] size N x dx 
        @param y        [nd.array] size N x dy 
        @param kwargs   [dic] params for KozLeoHestimator

    Outputs
        @param H    [float] conditional_entropy

    N -> number of samples
    d[x|y] -> number of dimensions for X|Y
    '''
    xy = np.hstack([ x, y ])

    hy  = KozLeoHestimator_nats(y, **kozleoargs)
    hxy = KozLeoHestimator_nats(xy, **kozleoargs)

    return hxy - hy,hy


def compute_flux(p):
    '''
    @brief compute the flux of information from N variables to another

    @details given the joint pdf of N+1 variables, compute all the possible
    fluxes combinations of the N variables to the +1 variable
        T = compute_flux(p)
    
    Parameters
        p:   np.array
            multi-dimensional array containing the pdfs of the variables. 
            The first dimension corresponds to the index of the variable:
                p[0]  -> target variable
                p[1:] -> input variables

    Returns 
        T:  dict
            all possible fluxes from input vars to target vars:
                T[j] -> flux from tuple j to variable.

    '''
    Np = len(p.shape) # size of p
    inds = range(1,Np)

    T = {}
    for i in inds:
        for j in list(icmb(inds,i)):
            #print(j)

            # index 0 in p is the future variable
            # index 1...N are the variables
            noi = tuple(set(inds)-set(j))
            Hc_j_noi = cond_entropy(p,(0,),noi)
            Hc_j_all = cond_entropy(p,(0,),inds)
            T[j] = Hc_j_noi - Hc_j_all

    #for i in inds:
    #    for j in list(icmb(range(1,Np),i)):
    #        print(f'computing {j}')
    #        for lj in range(len(j)):
    #            for k in list(icmb(j,lj)):
    #                if len(k):
    #                    print(f'   we substract {k}')
    #                    T[j] = T[j] - T[k]

    for i in inds:
        for j in list(icmb(range(1,Np),i)):
            print(f'computing {j}')
            # All the combinations that must be subtracted to T[j]
            lj = [list(icmb(j,k)) for k in range(len(j))][1:]
            # ichain makes the list of list lj a single list:
            T[j] -= sum([T[a] for a in list(ichain.from_iterable(lj))])

    return T


class pdf:

    def __init__(self,p=None):
        self.p = p

    def init_from_arr(self,V,norm=True,bins=10,lims=(0,1)):

        if norm:
            V = np.array([normVec(V[i],lims) for i in range(len(V))])
            hist,_ = np.histogramdd(V.T,bins=bins,range=[lims for _ in range(len(V))])
        else:
            hist,_ = np.histogramdd(V.T,bins=bins,range=lims)

        self.p = hist/hist.sum()

    def entropy(self,i):
        return entropy_nvars(self.p,i)

    def cond_entropy(self,i,j):
        return cond_entropy(self.p,i,j)

    def mutual_info(self,i,j):
        return mutual_info(self.p,i,j)

    def cond_mutual_info(self,i,j,k):
        return cond_mutual_info(self.p,i,j,k)

def uniqueflux(p: np.ndarray)-> tuple[dict, dict, dict]:
    '''Compute flux decomposition of time signals (Adrian's approach)
    
    Given a PMF of the target variable in the future (signal s) and the
    agent variables (signals a) in the present, compute a decomposition 
    of the mutual information I( S; A ) into redundacy and synergy 
    positive particles
    
        Tred, Tsyn, Ia = uniqueflux(p)

    Parameters
        p:  np.ndarray [ Nbins_s, Nbins_a1, ... Nbins_aN ]
            Probability mass function. First dimension is the target variable

    Returns
        Tred:   dict
            Dictionary with the redundancies for each variable combination

        Tsyn:   dict
            Dictionary with the synergies for each variable combination
        
        Ia:     dict
            Mutual information between the source and each possible variable 
            combination

    '''
        
    p += 1e-14      # mollify the pmf to avoid nans
    p /= p.sum()    # re-normalize the pmf

    Ntot = p.ndim           # Number of variables ( + target )
    Nvars = Ntot - 1        # Number of variables
    
    Nt = p.shape[0]         # Number of target states 

    inds = range(1,Ntot) # source variables indeces

    # Compute the marginal pdf of the target variable
    p_s = p.sum( axis= (*inds,), keepdims= True )

    combs = []  # list of possible variable combinations 
    Is = {}     # Init a dictionary with the specific mutual info for each comb
    for i in inds:                      # -> for a number of var: 1 var, 2 vars ...
        for j in list(icmb(inds,i)):    # -> compute posible combinations

            combs.append(j) # -> save list of cominations to init objects later
            noj = tuple(set(inds)-set(j)) # all vars but the current combination

            ## Compute specific information for each j combination
            p_a   = p.sum( axis= (0,*noj), keepdims = True )    # p( {aj} )
            p_as  = p.sum( axis= noj, keepdims = True )         # p(s, {aj} )

            p_a_s = p_as / p_s  # p( {aj}| s )
            p_s_a = p_as / p_a  # p(s | {aj} )
            
            # Each Is[j] is an array of Nbins_target values
            Is[j] = (p_a_s * ( mylog(p_s_a) - mylog(p_s) )).sum(axis=j).ravel()

    Ia = { k: (Is[k] * p_s.squeeze()).sum() for k in Is.keys() }

    Tred = {cc: 0 for cc in combs}          # Init red dict with all possible combs
    Tsyn = {cc: 0 for cc in combs[Nvars:]}  # Init syn dict with +1 variable combs

    # For each value of the target, sort the Is and compute the increments
    for t in range(Nt):

        # Get array with the Is of all combinations, for a single t
        I1 = np.array( [ii[t] for ii in Is.values()] )        
        
        # Sort the Is in increasing order, and store the combs to keep track 
        i1 = np.argsort( I1 )               # sorted indices
        lab = [combs[i_] for i_ in i1]      # combinations in sorted order
        lens = np.array([len(l) for l in lab])  # compute the number of vars for each tuple
        
        # Make Is = 0 all combinations that are smaller than the maximum Is with one
        # variable
        I1 = I1[i1] # sorted specific I
        for l in range(1,lens.max()):               # -> number of vars together
            inds_l2 = np.where( lens == l+1 )[0]    # combinations with l+1 number of vars
            Il1max  = I1[lens==l].max()       # Maximum Is of combinations with l vars 

            # Set  Is = 0 for those cases whose Is is smaller than Il1max
            inds_ = inds_l2[ I1[inds_l2] < Il1max ]   
            I1[inds_] = 0                                   
        
        # Once we have remove the terms smaller, sort Is and the labs  again, and compute
        # the jumps in Is (Di)

        # Sort the Is in increasing order, and store the combs to keep track 
        i1 = np.argsort( I1 )     # sorted indices
        lab = [lab[i_] for i_ in i1]    # combinations in sorted order

        Di = np.diff( I1[i1], prepend=0. )  # Is increment 

        red_vars = list(inds)   # init redundancy labels (all single variables)
        
        # Compute redundancy and syngergy
        for i_,ll in enumerate( lab ):  # -> for all combinations

            info = Di[i_] * p_s.squeeze()[t]    # compute Di[t] * ps[t]

            if len(ll) == 1:                    # -> Redundancies if one element
                Tred[tuple(red_vars)] += info
                red_vars.remove(ll[0])
            else:                               # -> Synergy if more than one element
                Tsyn[ll] += info

    return Tred, Tsyn, Ia

