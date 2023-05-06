from wang_mendel import wang_mendel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from T1_set import *
from T1_output import T1_Triangular_output, T1_RightShoulder_output, T1_LeftShoulder_output


def get_MSE(real_values_list, predicted_value_list):
        return(np.square(np.subtract(real_values_list, predicted_value_list)).mean())


def main():
    df = pd.read_csv('data/truck.csv').drop(['t'], axis=1)
    X = df.values
    
    granularisation = [9,9,11] # 7 Antecedents for input1, 5 Antecedents for input2, 7 Consequents for Output
    
    #Generating rules from noise free set
    train_obj = wang_mendel(X, granularisation)
    print(train_obj.reduced_rules)
    
    X_test = X[:,:2]
    output_results = train_obj.compute(X_test) # Compute outputs
    mse=get_MSE(X[:,-1], output_results)
    print('MSE:\n', mse)
    print(output_results)

    # Plot the results
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 15})
    
    plt.gca().yaxis.grid(True)
    plt.plot(X[:,-1], label='True')
    plt.plot(output_results, 'r-.', label='Pred')
    plt.title("Comparison\nMSE: "+str(round(mse,4)))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()