import numpy as np

#def distance(u,v,p):
#    return np.mean(abs(u-v)**p,axis = 1)
  
def wkmeans(X,k=2,p=1,threshold=1e-3,max_iter = 1000,random_state=None,verbose = True, k_means_pp = True):
    
    """
    X: M x N matrix with M log_ret time_series
    k: Number of clusters
    p: Wp-metric
    Return: (k x N matrix that represents k centroids, array of M labels)
    """
    
    X_sorted = np.sort(X,axis = 1)
    
    rng = np.random.default_rng(random_state)
    
    if (k_means_pp):
        barycenters_0 = X_sorted[rng.choice(X.shape[0])].reshape(1,-1)
        
        for i in range(k-1):
            min_d = np.array([distance(X_sorted,barycenters_0[i],p) for i in range(barycenters_0.shape[0])]).min(axis = 0)
            barycenters_0 = np.vstack([barycenters_0,X_sorted[min_d.argmax()]])                  

    else:
        barycenters_0 = X_sorted[rng.choice(X.shape[0],k,replace=False)]
        
        
    barycenters_1 = None
    cluster = None
    counter = 0
    

    while (counter <= max_iter):
        
        cluster = np.array([distance(X_sorted,barycenters_0[i],p) for i in range(k)]).argmin(axis = 0)
        
        if (p == 1):
            
            barycenters_1 = np.array([np.median(X_sorted[cluster == i], axis = 0) for i in range(k)])

        elif (p > 1):
            
            barycenters_1 = np.array([np.mean(X_sorted[cluster == i], axis = 0) for i in range(k)]) 
            
        counter += 1
        
        loss = np.sum([distance(X_sorted[cluster == i],barycenters_1[i],p).sum() for i in range(k)])
        
        if (verbose):
            
            print("Mean Loss after {} iterations: {}".format(counter,loss/len(X)))
        
        
        if (sum(distance(barycenters_0,barycenters_1,p)) < threshold):
        
            print("Convergence Reached! Number of Iterations: {}".format(counter))
            
            break
        
        else:
            
            barycenters_0 = barycenters_1.copy()
            

    if(counter == max_iter):
        
        print('Maximum Number of Iteration Reached!')
    
    return barycenters_1, cluster
