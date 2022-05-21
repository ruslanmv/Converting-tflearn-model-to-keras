'''
In ordering to simplify the discussion of this notebook let us define the following helper function, which simply give us the activation shape of the Convolution Network
'''
def nparameters_convolution(inputs, kernel,stride,padding,filters):
    '''
    input = (nh, nw, nc)
    where 
    nh: height
    nw: widht
    nc: channels
    
    activation_shape  = (nhl,nwl,ncl)
       
    nhl = (nh+2*p-fw)/s + 1
    nwl = (nw+2*p-fh)/s + 1
    ncl = filters
    
    
    where
       fw,fh : filter sizes
       p : padding
       s : stride  
    ncl  : filters
    
    '''
    nh,nw,nc = inputs
    fh,fw = kernel
     
    s        = stride
    p        = padding
    ncl      = filters
    
    #activation shape 
    nhl = (nh+2*p-fw)/s + 1
    nwl = (nw+2*p-fh)/s + 1
    activation_shape = (int(nhl),int(nwl),int(ncl))
    
    # activation size
    activation_size=int(nhl)*int(nwl)*int(ncl)
    
    
    # number Parameters
    nparameters=((fh*fw*nc)+1)*ncl
    
    print("Activation Shape,", "Activation Size,","# Parameters")
     
    return   activation_shape ,  activation_size, nparameters
    
#round a float up to next even number
import math
def roundeven(f):
    return math.ceil(f / 2.) * 2