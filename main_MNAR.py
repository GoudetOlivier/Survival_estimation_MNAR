from src.fn import *
import torch
from os import system
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import argparse
from src.neuralnets import NNetWrapper as nnetwrapper
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from scipy.stats import exponweib
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from joblib import Parallel, delayed
import warnings

import pandas as pd

from src.missingCensorshipModels import HeckMan_MAR, HeckMan_MNAR,  HeckMan_MNAR_two_steps, Linear, Neural_network_regression, WeibullMechanism
from itertools import product

##########
#  Main  #
##########
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data', metavar='d', type=str, help='data')
    
    parser.add_argument('model', metavar='d', type=str, help='data')
    
    parser.add_argument('--sample_size', metavar='d', type=int, help='sample_size', default=2000)
    parser.add_argument('--rho', metavar='d', type=float, help='rho', default=0.5)
    parser.add_argument('--pr_xi', metavar='d', type=int, help='pr_xi', default=50)
    parser.add_argument('--pr_delta', metavar='d', type=int, help='pr_delta', default=50)
    parser.add_argument('--seed', metavar='d', type=int, help='seed', default=0)

    args = parser.parse_args()
    
    seed = args.seed
    
    
    
    rho = args.rho
    
    pr_xi = args.pr_xi
    pr_delta = args.pr_delta

    #device = "cuda:" + str(int(num_device))
    device = "cpu"

    if("pros" in args.data):
        type_data = "pros"

    filename = args.data

    print(filename)
    
    if (type_data == "pros"):
        f_rho = open("rho " + str(filename) + ".txt", "a")
        


    nb_iter = 1

    sample_size = args.sample_size



    n_jobs_cross_val = 20

    x_list = [0.25, 0.5, 0.75]
    
    list_covariable_values = []

    for x1 in x_list:
        for x2 in x_list:
            for x3 in x_list:
                list_covariable_values.append(np.asarray([x1,x2,x3]))


    # list_bw_subramanian = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    # list_h = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

    list_bw_subramanian = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    list_h = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]

    # list_bw_subramanian = [0.1]
    # list_h = [0.3]

    lr = 0.001


    nb_epoch_xi = 100
    nb_epoch_delta_rho = 100
    nb_epoch_delta_MAR = 100
    batch_size = 100

    



    for type_data in [type_data]:

        for noCovariateMode in [  False ]:

            for ll in [1]:

                if (type_data == "pros"):

                    data = pd.read_csv("data/" + filename + ".csv", index_col=0)

                    if(args.model == "Subramanian"):

                        data=data.sample(frac=0.5,random_state=90)
                        print(data.shape)


                    T = np.expand_dims(data["T"].values, axis = 0)
                    

                    delta = np.expand_dims(data["delta"].values, axis=0)
                    xi = np.expand_dims(data["xi"].values, axis=0)



                    covariates = data[data.columns[:-3]].values
                    scaler = MinMaxScaler()

                    scaler.fit(covariates)


                    covariates = scaler.transform(covariates)

                    X = np.expand_dims(covariates, axis=0)
                    XS = np.expand_dims(covariates, axis=0)



                    sample_size = X.shape[1]

                    print("X shape")
                    print(X.shape)

                    list_covariable_values = []

                    list_values_all_covariables = []

                    for c in range(X[0].shape[1]):

                        list_values = []


                        for perc in [10,25,50,75,90]:

                            list_values.append(np.percentile(X[0][:,c], perc))



                        list_values_all_covariables.append(list_values)

                    print("list_values_all_covariables")
                    print(list_values_all_covariables)



                    all_configs = list(product(list_values_all_covariables[0],list_values_all_covariables[1],list_values_all_covariables[2]))

                    print("all_configs")
                    print(all_configs)

                    all_perc = list(product([10,25,50,75,90],[10,25,50,75,90],[10,25,50,75,90]))


                    print("len(all_configs)")
                    print(len(all_configs))

                    print("all_perc")
                    print(all_perc)

                    list_covariable_values = []

                    for config in all_configs:

                        list_covariable_values.append(np.array(config))


                    index = np.argsort(T, axis=1)

                    for i in range(nb_iter):
                        T[i, :] = T[i, index[i, :]]
                        delta[i, :] = delta[i, index[i, :]]
                        xi[i, :] = xi[i, index[i, :]]
                        X[i, :] = X[i, index[i, :]]
                        XS[i, :] = XS[i, index[i, :]]



                elif (type_data == "weibull"):

                    if(noCovariateMode):
                        a = [3, 0, 0, 0]
                        b = [2.5, 0, 0, 0]
                        c = [0.4, 0, -0.9]
                    else:
                        
                        if(pr_xi == 50 and pr_delta == 25):
                            
                            a = [3, 4, 3,-0.2]
                            b = [2.75, -2, 0.4,-0.5]
                            c = [-0.6,1,0.2,-0.1,0.5]
                            
                        elif(pr_xi == 50 and pr_delta == 50):
                            
                            a = [1, 4, 3,-0.2]
                            b = [0.9, -0.1, 0.4,-0.5]
                            c = [-0.7,1,0.2,-0.1,0.5]                            

                        elif(pr_xi == 50 and pr_delta == 75):
                            
                            a = [2, 4, 3,-0.2]
                            b = [0.2, -0.1, 0.2,0.1]
                            c = [-0.7,1,0.2,-0.1,0.5]   
                            
                        elif(pr_xi == 75 and pr_delta == 25):
                            
                            a = [3, 4, 3,-0.2]
                            b = [2.75, -2, 0.4,-0.5]
                            c = [0.1,1,0.2,-0.1,0.5]
                            
                        elif(pr_xi == 75 and pr_delta == 50):
                            
                            a = [1, 4, 3,-0.2]
                            b = [0.9, -0.1, 0.4,-0.5]
                            c = [0.1,1,0.2,-0.1,0.5]                      

                        elif(pr_xi == 75 and pr_delta == 75):
                            
                            a = [2, 4, 3,-0.2]
                            b = [0.2, -0.1, 0.2,0.1]
                            c = [0.1,1,0.2,-0.1,0.5]                                     

                        elif(pr_xi == 25 and pr_delta == 25):
                            
                            a = [3, 4, 3,-0.2]
                            b = [2.75, -2, 0.4,-0.5]
                            c = [-1.3,1,0.2,-0.1,0.5]  
                            
                        elif(pr_xi == 25 and pr_delta == 50):
                            
                            a = [1, 4, 3,-0.2]
                            b = [0.9, -0.1, 0.4,-0.5]
                            c = [-1.3,1,0.2,-0.1,0.5]  
                            
                        elif(pr_xi == 25 and pr_delta == 75):
                            
                            a = [2, 4, 3,-0.2]
                            b = [0.2, -0.1, 0.2,0.1]
                            c = [-1.4,1,0.2,-0.1,0.5]                              
                            
                    Y, C, T, delta, xi, X, XS, probaDelta, probaXi = test_gen_data_weibull_model_Heckman_Mnar(nb_iter, sample_size,a, b, c, rho, seed)
                    



                elif (type_data == "frechet"):

                    if (noCovariateMode):
                        a = [3, 0, 0, 0]
                        b = [0.5, 0, 0, 0]
                        c = [0.3, 0, -0.3]
                    else:

                        if(pr_xi == 50 and pr_delta == 25):
                            a = [2, 2, 0.2, 1]
                            b = [1.8, -0.1, 0.4, -0.5]
                            c = [-0.55, 1,0.2,-0.1, -0.3]
                            
                        if(pr_xi == 50 and pr_delta == 50):
                            a = [2, 2, 0.2, 1]
                            b = [0.9, -0.1, 0.4, -0.5]
                            c = [-0.55, 1,0.2,-0.1, -0.3]
                            
                        if(pr_xi == 50 and pr_delta == 75):
                            a = [2, 2, 0.2, 1]
                            b = [0.2, -0.1, 0.25, 0.1]
                            c = [-0.6, 1,0.2,-0.1, -0.3]
                            
                            
                        if(pr_xi == 75 and pr_delta == 25):
                            a = [2, 2, 0.2, 1]
                            b = [1.8, -0.1, 0.4, -0.5]
                            c = [0.1, 1,0.2,-0.1, -0.3]
                            
                        if(pr_xi == 75 and pr_delta == 50):
                            a = [2, 2, 0.2, 1]
                            b = [0.9, -0.1, 0.4, -0.5]
                            c = [0.1, 1,0.2,-0.1, -0.3]
                            
                        if(pr_xi == 75 and pr_delta == 75):
                            a = [2, 2, 0.2, 1]
                            b = [0.2, -0.1, 0.25, 0.1]
                            c = [0.1, 1,0.2,-0.1, -0.3]
                            
                            
                        if(pr_xi == 25 and pr_delta == 25):
                            a = [2, 2, 0.2, 1]
                            b = [1.8, -0.1, 0.4, -0.5]
                            c = [-1.3, 1,0.2,-0.1, -0.3]
                            
                        if(pr_xi == 25 and pr_delta == 50):
                            a = [2, 2, 0.2, 1]
                            b = [0.9, -0.1, 0.4, -0.5]
                            c = [-1.4, 1,0.2,-0.1, -0.3]
                            
                        if(pr_xi == 25 and pr_delta == 75):
                            a = [2, 2, 0.2, 1]
                            b = [0.2, -0.1, 0.25, 0.1]
                            c = [-1.4, 1,0.2,-0.1, -0.3]                          
                            

                    Y, C, T, delta, xi, X, XS, probaDelta, probaXi = test_gen_data_frechet_model_Heckman_Mnar(nb_iter,
                                                                                                            sample_size,
                                                                                                            a, b, c,
                                                                                                            rho, seed)


                print("frac xi 1")
                print(np.sum(xi[0])/sample_size)
                print("frac delta 1")
                print(np.sum(delta[0])/sample_size)
                print("th.sum(delta * xi)")
                print(np.sum(delta[0] * xi[0])/sample_size)
                print("th.sum((1-delta) * xi)")
                print(np.sum((1 - delta[0]) * xi[0])/sample_size)
                print("th.sum(delta * (1-xi))")
                print(np.sum(delta[0] * (1-xi[0]))/sample_size)
                print("th.sum((1-delta) * (1-xi))")
                print(np.sum((1 - delta[0]) * (1-xi[0]))/sample_size)

                if(type_data != "pros"):
                    list_Y_obs = [ Y[i,xi[i,:] == 1]  for i in range(nb_iter) ]
                    list_C_obs = [C[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_T_obs = [T[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_delta_obs = [delta[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_X_obs = [X[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_XS_obs = [XS[i, xi[i, :] == 1] for i in range(nb_iter)]

                if (type_data != "pros"):
                    list_probaDelta_obs = [probaDelta[i, xi[i, :] == 1] for i in range(nb_iter)]
                    list_probaXi_obs = [probaXi[i, xi[i, :] == 1] for i in range(nb_iter)]

                if (X[0].ndim == 2):
                    d = X[0].shape[1]
                else:
                    d = 1

                if (XS[0].ndim == 2):
                    dS = XS[0].shape[1]
                else:
                    dS = 1

                if (noCovariateMode):
                    d = 0
                    dS = 0


                dict_p = {}


                #"Subramanian","NN_two_steps", "NN_MAR" 

                list_model = [ args.model ]


                
                for type_model in list_model:

                    p = np.zeros((nb_iter, sample_size))

                    if (type_model == "Standard_beran"):

                        p = delta

                    elif (type_model == "True_proba"):

                        p = probaDelta


                    elif (type_model == "Linear" or type_model == "NN" or type_model == "NN_two_steps"  or type_model == "sameClass"):

                        if(type_model == "Linear"):

                            f = Linear(d+1)
                            g = Linear(dS+1)



                        elif(type_model == "NN" or type_model == "NN_two_steps" ):


                            if(noCovariateMode):
                                f = Neural_network_regression([d + 1, 200, 200, 100, 1])
                                g = Neural_network_regression([dS + 1, 200, 200, 100, 1])
                                # f = Neural_network_regression([d + 1, 200, 100, 1])
                                # g = Neural_network_regression([dS + 1, 200, 100, 1])
                            else:
                                f = Neural_network_regression([d + 1, 200, 200, 100, 1])
                                g = Neural_network_regression([dS + 1, 200, 200, 100, 1])

                        elif (type_model == "sameClass"):
                            
                            if (type_data == "weibull"):
                                f = WeibullMechanism(d)
                                g = Linear(dS + 1)



                        for k in range(nb_iter):

                            relaunch = True
                            
                            print("NN_MNAR")
                            
                            
                            while (relaunch == True):

                                # if(type_model == "NN_two_steps"):

                                if (type_data == "pros"):
                                    hMnar = HeckMan_MNAR_two_steps(f, g, device, noCovariateMode, None)
                                else:
                                    hMnar = HeckMan_MNAR_two_steps(f, g, device, noCovariateMode, rho)

                                print("fit xi")
                                if (type_data == "pros"):
                                    hMnar.fit_xi(XS[k,], T[k,], xi[k,], None, lr, nb_epoch_xi, batch_size)
                                else:
                                    hMnar.fit_xi(XS[k,], T[k,], xi[k,], probaXi[k,], lr, nb_epoch_xi, batch_size)


                                print("fit delta rho")
                                if (type_data == "pros"):
                                    hMnar.fit_delta_rho(X[k,], XS[k,], T[k,], delta[k,], xi[k,], None, lr, lr ,nb_epoch_delta_rho, batch_size)
                                else:
                                    hMnar.fit_delta_rho(X[k,], XS[k,], T[k,], delta[k,], xi[k,], probaDelta[k,], lr, lr,
                                                        nb_epoch_delta_rho, batch_size)

                                p[k, :] = hMnar.predict(X[k,], T[k,])



                                
                                
                                if (np.isnan(np.sum(p[k, :])) == False):

                                    relaunch = False
                                else:
                                    nb_epoch_delta_rho = nb_epoch_delta_rho//2
                                    nb_epoch_xi = nb_epoch_xi//2



                                # else:
                                #
                                #     print("NN_together")
                                #     hMnar = HeckMan_MNAR(f, g, device, noCovariateMode)
                                #     hMnar.fit(X[k,], XS[k,], T[k,], delta[k,], xi[k,], probaDelta[k,],lr1, lr2 ,nb_epoch_delta_rho,batch_size)

                            rho = hMnar.getRho()
                                
                            f_rho.write(str(seed) + "," + str(rho) + "\n")
                            f_rho.close()


                    elif (type_model == "Linear_MAR" or type_model == "NN_MAR" or type_model == "sameClass_MAR"):


                        if (type_model == "Linear_MAR"):

                            f = Linear(d+1)

                        elif (type_model == "NN_MAR"):

                            if(noCovariateMode):
                                f = Neural_network_regression([d + 1, 200, 200, 100, 1])
                            else:
                                f = Neural_network_regression([d + 1, 200, 200, 100, 1])


                        for k in range(nb_iter):

                            relaunch = True

                            print("NN_MAR")
                            
                            while(relaunch):

                                mnar = HeckMan_MAR(f, device, noCovariateMode)

                                if (type_data == "pros"):
                                    mnar.fit(list_X_obs[k],  list_T_obs[k], list_delta_obs[k], None, lr, nb_epoch_delta_MAR, batch_size)
                                else:
                                    mnar.fit(list_X_obs[k], list_T_obs[k], list_delta_obs[k], list_probaDelta_obs[k],
                                             lr, nb_epoch_delta_MAR, batch_size)

                                p[k, :] = mnar.predict(X[k,], T[k,])

                                if(np.isnan(np.sum(p[k, :]))==False):                              
                                    relaunch = False                  
                                else:    
                                    nb_epoch_delta_MAR = nb_epoch_delta_MAR//2
                                    


                    if (type_model == "Linear_with_delta"):
                        dict_p["Linear_with_delta"] = dict_p["Linear"] * (1-xi) + delta * xi
                    elif(type_model == "NN_with_delta"):
                        dict_p["NN_with_delta"] = dict_p["NN"] * (1 - xi) + delta * xi
                    elif(type_model == "Linear_MAR_with_delta"):
                        dict_p["Linear_MAR_with_delta"] = dict_p["Linear_MAR"] * (1 - xi) + delta * xi
                    elif(type_model == "NN_MAR_with_delta"):
                        dict_p["NN_MAR_with_delta"] = dict_p["NN_MAR"] * (1 - xi) + delta * xi

                    elif (type_model == "NN_two_steps_with_delta"):
                        dict_p["NN_two_steps_with_delta"] = dict_p["NN_two_steps"] * (1 - xi) + delta * xi
                    else:
                        dict_p[type_model] = p

                    print("1 - xi")
                    print(np.mean(1 - xi))
                    print(1 - xi)

                    print("delta")
                    print(delta)

                    print("delta * xi")
                    print(delta * xi)
                    print(np.mean(delta * xi))





                    print("type_model")
                    print(type_model)
                    print("np.min(p)")
                    print(np.min(dict_p[type_model]))

                    print("np.max(p)")
                    print(np.max(dict_p[type_model]))

                    print("p")
                    print(dict_p[type_model])

                #######################
                # Survival estimation #
                #######################
                print('Survival estimators computing')

                dict_beran = {}

                num_t = 100

                if (type_data != "pros"):
                    t = np.linspace(np.amin(Y), np.amax(Y), num=num_t)
                else:
                    t = np.linspace(0, 1, num=num_t)

                for type_model in list_model:

                    print(type_model)

                    beran = np.zeros((nb_iter, len(t), len(list_covariable_values)))

                    pbar = tqdm(range(nb_iter))

                    p = dict_p[type_model]


                    if(noCovariateMode == False):
                        if (type_model == "Standard_beran"):
                            list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                                delayed(cross_val_beran)(T.shape[1], T[k, :], delta[k, :], p[k, :],
                                                         X[k, :], list_h, k) for k in
                                range(nb_iter))

                        elif (type_model == "Subramanian"):

                            print("OK OK ")
                            # cross_val_beran_Subramanian(n, obs, delta, xi, x, list_h1, list_h2, k):

                            list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                                delayed(cross_val_beran_Subramanian_beran)(T.shape[1], T[k, :], delta[k, :], xi[k, :],
                                                         X[k, :], list_bw_subramanian, list_h, k) for k in
                                range(nb_iter))

                        else:
                            list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                                delayed(cross_val_beran_proba)(T.shape[1], T[k, :], p[k, :], X[k, :],
                                                               list_h, k) for k in
                                range(nb_iter))

                    else:

                        if (type_model == "Subramanian"):

                            list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                                delayed(cross_val_beran_Subramanian_Kaplan)(T.shape[1], T[k, :], delta[k, :], xi[k, :],
                                                                           X[k, :], list_h,  k) for k in
                                range(nb_iter))

                            print("list_best_h")
                            print(list_best_h)



                    for k in pbar:


                        if (type_model == "Subramanian"):

                            if(noCovariateMode):
                                p[k, :] = Subramanian_estimator_no_covariate(T[k,], delta[k,], xi[k,], list_best_h[k])
                            else:
                                p[k, :] = Subramanian_estimator(X[k,], T[k,], delta[k,], xi[k,],list_best_h[k][0] )

                        c_x = 0
                        for x_eval in list_covariable_values:

                            if(noCovariateMode):
                                beran[k, :, c_x] = beran_estimator(t, T[k, :], p[k, :], X[k, :], x_eval, -1, False,
                                                                   True)
                            else:
                                if (type_model == "Subramanian"):
                                    beran[k, :, c_x] = beran_estimator(t, T[k, :], p[k, :], X[k, :], x_eval, list_best_h[k][1],
                                                                       False, False)
                                else:
                                    beran[k, :, c_x] = beran_estimator(t, T[k, :], p[k, :], X[k, :], x_eval, list_best_h[k], False, False)

                            c_x += 1

                    dict_beran[type_model] = beran

                    #np.save("save/" + type_model, beran)




                if(args.model == "Standard_Beran_delta_obs_only"):
                    
                    #Test beran on observed delta
                    print("Standard_Beran_delta_obs_only")
                    beran = np.zeros((nb_iter, len(t), len(list_covariable_values)))

                    if (noCovariateMode == False):
                        list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                            delayed(cross_val_beran)(list_T_obs[k].shape[0], list_T_obs[k], list_delta_obs[k], list_delta_obs[k],
                                                    list_X_obs[k], list_h, k) for k in range(nb_iter))

                    for k in pbar:
                        c_x = 0
                        for x_eval in list_covariable_values:
                            if (noCovariateMode == False):
                                beran[k, :, c_x] = beran_estimator(t, list_T_obs[k], list_delta_obs[k], list_X_obs[k], x_eval,
                                                                list_best_h[k], False, False)

                            else:
                                beran[k, :, c_x] = beran_estimator(t, list_T_obs[k], list_delta_obs[k], list_X_obs[k], x_eval, -1, False, True)

                            c_x += 1
                    dict_beran["Standard_Beran_delta_obs_only"] = beran
                    list_model.append("Standard_Beran_delta_obs_only")




                #######################
                # Compute results     #
                #######################

                df_results_mise = pd.DataFrame()

                if (type_data != "pros"):
                    true_cdf = np.zeros((num_t, len(list_covariable_values)))

                df_results_mise["t"] = t

                for i in range(len(list_covariable_values)):

                    if (type_data != "pros"):
                        if (type_data == "weibull"):

                            true_cdf[:, i] =  scipy.stats.weibull_min(a[0] + a[1] * list_covariable_values[i][0] + a[2] * list_covariable_values[i][1] + a[3] * list_covariable_values[i][2]).cdf(t)

                        elif (type_data == "frechet"):

                            true_cdf[:, i] =  scipy.stats.frechet_r(a[0] + a[1] * list_covariable_values[i][0] + a[2] * list_covariable_values[i][1] + a[3] * list_covariable_values[i][2]).cdf(t)

                    if (type_data != "pros"):
                        df_results_mise["true_cdf_" + str(list_covariable_values[i][0]) + "," + str(list_covariable_values[i][1]) + "," + str(list_covariable_values[i][2])] = true_cdf[:, i]

                    for idx, type_model in enumerate(list_model):

                        beran = dict_beran[type_model]
                        mean_beran = np.mean(beran[:, :, i], axis=0)

                        str_ = type_model + "_cdf_"
                        for j in range(X.shape[2]):
                            str_ += str(all_perc[i][j]) + "-"

                        df_results_mise[str_] = mean_beran

                        if (type_data != "pros"):
                            mise_beran = np.mean((beran[:, :, i] - true_cdf[:, i]) ** 2, axis=0)

                            df_results_mise[type_model + "_mise_" + str(list_covariable_values[i][0]) + "," + str(list_covariable_values[i][1]) + "," + str(list_covariable_values[i][2])] = mise_beran

                if (type_data == "pros"):
                    df_results_mise.to_csv("results/" + args.data + "/cdf_" + "pros" + "_" + str(sample_size) + "_" + args.model + "_seed_" + str(seed) + ".csv")


                if (type_data != "pros"):

                    f = open("results/" + type_data + "/" + "xi_" + str(pr_xi) + "_delta_" + str(pr_delta) + "/results_frechet_n_" + str(sample_size) + "_rho_" + str(rho) + "_xi_" + str(pr_xi) + "_delta_" + str(pr_delta) + "_seed_" + str(seed), "w")
                    str_ = "Model,"
                    for idx, type_model in enumerate(list_model):
                        str_ += type_model + ","
                    f.write(str_ + "\n")


                    for i in range(len(list_covariable_values)):
                        str_ = ""

                        str_ = "(" + str(list_covariable_values[i][0]) + "-" + str(list_covariable_values[i][1]) + "-" + str(list_covariable_values[i][2]) + "),"

                        for idx, type_model in enumerate(list_model):
                            beran = dict_beran[type_model]
                            str_ += str(np.mean((beran[:, :, i] - true_cdf[:, i]) ** 2)) + ","

                        f.write(str_ + "\n")

                    str_ = "Global score,"

                    for idx, type_model in enumerate(list_model):
                        beran = dict_beran[type_model]
                        str_ += str(np.mean((beran[0, :, :] - true_cdf) ** 2)) + ","

                    f.write(str_ + "\n")
                    f.close()

                #else:

                    #f = open("results/" + args.data + "/pros_comparison_MAR_MNAR " + "_seed_" + str(seed), "w")
                    #list_model = ["Standard_Beran_delta_obs_only", "Subramanian", "NN_two_steps", "NN_MAR"]
                    #list_model = ["Standard_Beran_delta_obs_only",  "NN_two_steps", "NN_MAR"]
                    #all_pairs = []

                    #for model1 in list_model:
                        #for model2 in list_model:
                           #if(model1 != model2 and model1 < model2):
                               #all_pairs.append([model1,model2])


                    #str_ = "age,pack_years,bmi_curr,"
                    #for pair  in all_pairs:
                        #str_ += "(" + pair[0] + "-" + pair[1] + ")" + ","
                    #f.write(str_ + "\n")

                    #for i in range(len(list_covariable_values)):
                        #str_ = ""
                        #for j in range(X.shape[2]):
                            #str_ += str(all_perc[i][j]) + ","

                        #for pair  in all_pairs:
                            #beran0 = dict_beran[pair[0]]
                            #beran1 = dict_beran[pair[1]]

                            #str_ += str(np.mean(np.abs(beran0[:, :, i] - beran1[:, :, i]))) + ","

                        #f.write(str_ + "\n")


                    #f.close()
