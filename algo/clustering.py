import numpy as np

def wkmeans(X,k=2,p=1,threshold=1e-3,max_iter = 1000,max_retries = 5):
    
    """
    X: M x N matrix with M log_ret time_series
    k: Number of clusters
    p: Wp-metric
    Return: (k x N matrix that represents k centroids, array of M labels)
    """
    
    X_sorted = np.sort(X,axis = 1)
    
    
    for trial in range(max_retries):
        
        rng = np.random.default_rng(None)
    
        barycenters_0 = X_sorted[rng.choice(X.shape[0],k,replace=False)]    
        barycenters_1 = np.empty((k,X.shape[1]))
    
    
        for counter in range(max_iter):
            
            break_out_flag = False
        
            clusters = np.array([np.mean(abs(X_sorted-barycenters_0[i])**p,axis = 1) for i in range(k)]).argmin(axis = 0)
            # This cannot be written in the same for loop as the updated barycenters depend on the current clusters
        
            for i in range(k):
                
                if (sum(clusters == i) == 0):
                    
                    break_out_flag = True
                    
                    break
                
        
                if (p == 1):
            
                    barycenters_1[i] = np.median(X_sorted[clusters == i], axis = 0)
                
                

                elif (p > 1):
            
                    barycenters_1[i] = np.mean(X_sorted[clusters == i], axis = 0)
                
            if (break_out_flag == True):
                
                print('Empty cluster found, retrying again.')
                
                break
            

            loss = np.sum(abs(barycenters_0 - barycenters_1)**p) / X.shape[1]
        
            print("Loss after {} iterations: {}".format(counter+1,loss))
        

            if (loss < threshold):
        
                print("Convergence Reached! Number of Iterations: {}".format(counter+1))
            
                return barycenters_1, clusters
        
            else:
            
                barycenters_0 = barycenters_1.copy()
            

        print('Maximum Number of Iteration Reached! Retrying again!')
    
        
    
    print('Clustering fails after {} retries'.format(max_retries))



