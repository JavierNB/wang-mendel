import numpy as np
from operator import mul
from T1_set import *


class wang_mendel(object):
    
    def __init__(self, train_data, antecedent_number):

        self.train_data=train_data
        self.antecedents=self.generate_antecedents(train_data, antecedent_number)
        self.__reduced_rules=self.rule_matrix_generating()
    
    # Step 1 - Divide the Input and Output Spaces into Fuzzy Regions
    def generate_antecedents(self,train_data,antecedent_number):
            antecedents = {}
            for ant, ant_n in zip(range(train_data.shape[1]), antecedent_number):
                antecedent = {}
                # UoD of current Input
                max_number = max(train_data[:, ant])
                min_number = min(train_data[:, ant])
                step = ((max_number - min_number) / (ant_n - 1)) / 4.0

                for i in range(1, ant_n + 1):
                    mean = min_number + (i - 1) * step * 4.0
                    if(i == 1):
                         antecedent[i] = T1_LeftShoulder(mean, step , 500)
                    elif(i == ant_n):
                        antecedent[i] = T1_RightShoulder(mean, step , 500)
                    else:
                        antecedent[i] = T1_Triangular(mean, step , 500)
                antecedents[ant]=antecedent
            return (antecedents)
    
    
    # Step 2 - Generate Fuzzy Rules from Given Data Pairs
    def rule_matrix_generating(self):
        # For each training record return the x value, membership and number of antecedent with the highest membership
        x_vals_with_mf= self.assign_points()
        all_rule_matrix = np.zeros([len(self.train_data), (self.train_data.shape[1] + 1)])
        for i in range(0, len(self.train_data)):
            rule_degree = 1.0
            for c in range(self.train_data.shape[1]):
                all_rule_matrix[i,c] = x_vals_with_mf[c][i][2] # Get number of antecedent fired
                rule_degree = x_vals_with_mf[c][i][1]*rule_degree # multiplication of degrees
            
            all_rule_matrix[i,-1]=rule_degree #Step 3 -Assign a Degree to each rule
        return(self.rule_reduction(all_rule_matrix))
    

    def assign_points(self):
        x_vals_with_mf = {}
        for c in range(self.train_data.shape[1]):
            x_val_with_mf = np.empty([len(self.train_data), (3)])
            for index, x in enumerate(self.train_data[:, c]):
                x_val_with_mf[index][0]=x
                x_val_with_mf[index][1:3]=self.get_antIndex_and_maxDegree(x,c)
            x_vals_with_mf[c] = x_val_with_mf
        return(x_vals_with_mf)
       
    def get_antIndex_and_maxDegree(self, x, c):
        max_degree=0.0
        for i in self.antecedents[c]:
            degree=self.antecedents[c][i].get_degree(x)
            if(degree > max_degree):
                max_degree=degree
                antIndex=i
    
        if(max_degree==0.0):
            raise ValueError( "There is no max degree")
        else:
            return ((max_degree,antIndex))    
    
    
    def rule_reduction(self,all_rule_matrix): #Accept the rule from a conflict group that has maximum degree
        for i in range(0, len(all_rule_matrix)):
            temp_rule_1 = all_rule_matrix[i]
            if not np.isnan(temp_rule_1).any() :
                for t in range(i + 1, len(all_rule_matrix)):
                    temp_rule_2 = all_rule_matrix[t]
                    # check antecedent equality
                    if np.array_equal(temp_rule_1[0:-2], temp_rule_2[0:-2]):
                        # check degree and keep the greatest
                        if(temp_rule_2[-1] > temp_rule_1[-1]):
                            # the rule, with lower degree, is replaced by the one with higher degree
                            all_rule_matrix[i] = all_rule_matrix[t]
                        all_rule_matrix[t] = np.nan
        # Step 4 - Create a Combined Fuzzy Rule Base
        return(all_rule_matrix[~np.isnan(all_rule_matrix).any(axis=1)])


    def individual_rule_output(inputs, rule):
        firing_level_of_pairs = 1        
        for i in range(0, len(inputs)):
            temp_firing = inputs[i][int(rule[i]) - 1]
            
            if(temp_firing == 0):
                firing_level_of_pairs = "nan"
                break
            # minimum is implemented
            if(temp_firing < firing_level_of_pairs):
                firing_level_of_pairs = temp_firing
            
        return firing_level_of_pairs


    # Step 5 - Determine a Mapping Based on the Combined Fuzzy Rule Base
    # Dont use yet!!!
    def compute(self, x_test):
        output_results = []
        for input_index, i in enumerate(x_test):
            rule_output_strength = np.empty([len(self.reduced_rules), 1])
            rule_output_strength.fill(np.NaN)
            # Here
            for rule_index, rule in enumerate(self.reduced_rules):
                
                for i in self.antecedents:
                    fs = self.antecedents[i].get_degree(input_val)
                    temp_firings.append(fs) # 1 dim, 7 values
                rule_output_strength[rule_index] = individual_rule_output(all_firing_strengts[input_index:(input_index + train_obj.p)], rule[0:train_obj.p])
        
        return 0
                       
    @property
    def mf_interval_matrix(self):
        return self.__mf_interval_matrix
    
    @property
    def reduced_rules(self):
        return self.__reduced_rules
    
    @property
    def shape(self):
        return self._shape

  

#np.savetxt(" .csv",self.reduced_rules,delimiter=",")