import numpy as np


class T1_sets(object):
    
    
    def __init__(self, peak, step, discretisation):
        
        
        self._peak=peak
        self._step=step
        self._interval= (peak - (4.0*step), peak + (4.0 * step) ) 
        self._discrete=discretisation
        

   
    @property
    def peak(self):
        return self._peak 
   

    @property
    def interval(self):
        return self._interval
    
    
    @property
    def step(self):
        return self._step
    
    
class T1_Triangular(T1_sets):
    
    
    def get_degree(self, x):
        
        left=self._interval[0]
        right=self._interval[1]
         
        if(x <= left or x>= right):
            degree=0.0
        elif(x < self._peak ):
            degree = (x - left) / (self._peak - left)       
        elif(x > self._peak ):
            degree = (right - x) / (right - self._peak)
        elif(x ==self._peak):
            degree = 1.0
            
        return(degree)   
    
    def get_mf_degrees(self):
        list_of_mf=[]
        list_of_x=np.linspace(self._interval[0],self._interval[1], self._discrete)
        
        for i in list_of_x:
            list_of_mf.append(self.get_degree(i))
        list_of_mf=np.asarray(list_of_mf)
        return(list_of_mf)

    def get_peak(self):
        return self._peak  
    

class T1_Gaussian(T1_sets):
    
    def get_degree(self, x):
        if(x>=self._interval[0] and x<=self._interval[1]):
            return np.exp(-np.power(x - self.peak, 2.) / (2*np.power(self.step, 2.)))    
        else:
            return 0.0   
    
    def get_mf_degrees(self):
        list_of_mf=[]
        list_of_x=np.linspace(self._interval[0],self._interval[1], self._discrete)
        
        for i in list_of_x:
            list_of_mf.append(self.get_degree(i))
        list_of_mf=np.asarray(list_of_mf)
        return(list_of_mf)
    

class T1_RightShoulder(T1_sets):
    
    
    
    def get_degree(self, x, maxDegree=1.0):
        left=self._interval[0]
        right=self._interval[1]
        
        if(x<left):
            return(0.0)
        elif(x>=self.peak):
             return(1)
        elif(x<self.peak and x>=left):
            return ( ((x-left)/float(self.peak - left)) )
        else:
            raise ValueError("Something wrong with x in Right Shoulder.")    
    
    def get_mf_degrees(self):
        list_of_mf=[]
        list_of_x=np.linspace(self._interval[0],self._interval[1], self._discrete)
        
        for i in list_of_x:
            list_of_mf.append(self.get_degree(i))
        list_of_mf=np.asarray(list_of_mf)
        return(list_of_mf)
    
    def get_peak(self):
        return self._peak 


class T1_LeftShoulder(T1_sets):
    
    def get_degree(self, x):
        left=self._interval[0]
        right=self._interval[1]
        
        if(x>right):
            return(0.0)
        elif(x<=self.peak):
            return(1.0)
        elif(x>self.peak and x<=right):
            return ( ((right-x)/float(right-self.peak))  )
        else:
            raise ValueError("Something wrong with x in Left Shoulder.")    
    
    
    def get_mf_degrees(self):
        list_of_mf=[]
        list_of_x=np.linspace(self._interval[0],self._interval[1], self._discrete)
        
        for i in list_of_x:
            list_of_mf.append(self.get_degree(i))
        list_of_mf=np.asarray(list_of_mf)
        return(list_of_mf)
    
    def get_peak(self):
        return self._peak 
        