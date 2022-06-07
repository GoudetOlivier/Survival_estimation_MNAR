import plotly.graph_objs as go
import plotly.offline as offline
# offline.init_notebook_mode(connected = True)
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('dataset', metavar='d', type=str, help='data')
parser.add_argument('pr_xi', metavar='d', type=int, help='data')
parser.add_argument('pr_delta', metavar='d', type=int, help='data')
    
args = parser.parse_args()

list_model = ["True_proba", "Standard_beran", "Standard_Beran_delta_obs_only",  "Subramanian",  "NN_MAR", "NN_two_steps", "NN_two_steps_with_delta"]
list_name = [ "Oracle", "Standard Beran", "Standard Beran", "MAR - Subramanian type", "MAR - Neural Network", "MNAR - Neural Network", "MNAR - Neural Network - with delta when observed" ]

list_legendGroup = ["Before data deletion","Before data deletion", "Only with observed delta" ,  "Naive MAR estimators", "Naive MAR estimators", "MNAR estimators" , "MNAR estimators"]

list_color = ['royalblue', 'orange', 'red',  'brown',  'yellow', 'green', 'cyan']


dico = {}
for model in list_model:
    dico[model] = []


dataset = args.dataset

pr_xi = args.pr_xi
pr_delta = args.pr_delta

x = []

for rho in ["0.0","0.25","0.5", "0.75"]:


    x.extend(r'$\large{\rho=' + rho + '}$' for i in range(200))


    cpt = 0

    #for filename in os.listdir("results/" + dataset + "/xi_" + str(pr_xi) + "_delta_" + str(pr_delta)):
    
        
        #print(filename)
        #if("Global_score_MAR001weibull_n_2000_nbIter_1_rho_" + str(rho) in filename):
            #print("ok")
            
            #file_oldname = filename
            #file_newname_newfile = "results_weibull_n_2000_rho_" + str(rho) + "_xi_" + str(pr_xi) + "_delta_" + str(pr_delta) + "_seed_" + str(cpt)
            #print(filename)
            #print(file_newname_newfile)
            
            #os.rename("results/" + dataset + "/xi_" + str(pr_xi) + "_delta_" + str(pr_delta) + "/" + file_oldname, "results/" + dataset + "/xi_" + str(pr_xi) + "_delta_" + str(pr_delta) + "/" + file_newname_newfile)
            
            #cpt += 1
            
    for seed in range(200):

        fileName = "results/" + dataset + "/xi_" + str(pr_xi) + "_delta_" + str(pr_delta) +  "/" + "results_" + dataset + "_n_2000_rho_" + str(rho) + "_xi_" + str(pr_xi) + "_delta_" + str(pr_delta) + "_seed_" + str(cpt)

        print(fileName)
        
        if(os.path.exists(fileName)):
            results = pd.read_csv(fileName)
            
            results = results.iloc[:, :-1]
            
            results = results.iloc[-1]

            if(cpt == 0):
                df = results
            else:
                df = df.append(results)

            
            cpt+=1


    for model in list_model:

        dico[model].extend(list(df[model].values))


data = []
for idx, model in enumerate(list_model):

    data.append(go.Box(
        y=dico[model],
        x=x,
        legendgroup=list_legendGroup[idx],
        legendgrouptitle_text=list_legendGroup[idx],
        name=list_name[idx],
        marker=dict(
            color=list_color[idx]
        )
    ))


# trace1 = go.Box(
#     y=[0.6, 0.7, 0.3, 0.6, 0.0, 0.5, 0.7, 0.9, 0.5, 0.8, 0.7, 0.2],
#     x=x,
#     name='radishes',
#     marker=dict(
#         color='#FF4136'
#     )
# )
# trace2 = go.Box(
#     y=[0.6, 0.7, 0.3, 0.6, 0.0, 0.5, 0.7, 0.9, 0.5, 0.8, 0.7, 0.2],
#     x=x,
#     name='carrots',
#     marker=dict(
#         color='#FF851B'
#     )
# )

# data = [trace0, trace1, trace2]

title = "MISE"

layout = go.Layout(
    yaxis=go.layout.YAxis(
        title=title,
        range=[0,0.02]
        #zeroline=False
    ),
    font=dict(
        family="Courier New, monospace",
        size=24
    ),
    boxmode='group',
    title= dataset + " - Pourcentage xi=1 : " + str(pr_xi) + "%" + " - Pourcentage delta=1 : " + str(pr_delta) + "%"
)
    



fig = go.Figure(data=data, layout=layout)

#fig.write_image("plots/boxplot_" + dataset + "_xi_" + str(pr_xi) + "_delta_" + str(pr_delta)  + ".png")


fig.show()

