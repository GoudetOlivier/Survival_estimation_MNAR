import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv("pros_data_mar22_d032222.csv")

# print(data["mortality_exitstat"].mean())

df_pros = data.loc[data["pros_cancer"]==1]

center = 8

df_pros = df_pros.loc[df_pros["center"]==center]



# df_pros = data.loc[(data["mortality_exitstat"]==3) | (data["mortality_exitstat"]==4)]
# print(df_pros)



# print("mortality_exitstat")
df_dummies_mortality_exitstat = pd.get_dummies(df_pros["mortality_exitstat"])

# print(df_dummies_mortality_exitstat.columns)
df_dummies_mortality_exitstat["delta"] = df_dummies_mortality_exitstat.iloc[:,0]
df_dummies_mortality_exitstat["xi"] = (df_dummies_mortality_exitstat.iloc[:,0] + df_dummies_mortality_exitstat.iloc[:,1])

print(df_dummies_mortality_exitstat["xi"] )
print("df_dummies_mortality_exitstat[xi].mean()")
print(df_dummies_mortality_exitstat["xi"].mean())

# age, sex

random.seed(0)


#fstcan_exitdays



list_all_variables = [("age","N"),("pack_years","N"),("bmi_curr","N")]

#pros_annyr


# list_variables_treatments = [("primary_trtp","C"),
#                   ("curative_hormp", "B"),
#                   ("curative_othp", "B"),
#                   ("curative_prostp", "B"),
#                   ("curative_radp", "B"),
#                   ("neoadjuvantp", "B")]
#
# list_variables_cancer_characteristics = [("pros_stage","C"),("pros_stage_m","C"),("pros_stage_n","C"),("pros_stage_t","C")]
#
#
# list_variables_demographics = [("age","N"),("sex","B"), ("educat","C"), ("marital","C"),("occupat","C")]
#
#
# list_variables_smoking = [("cig_stat","C"), ("cigpd_f","N"),("cigar","C")]
#
# list_variables_family = [("fh_cancer","B"), ("pros_fh","C")]
#
# list_variables_body = [("bmi_curr","N"), ("weight_f","N")]


# list_all_variables = []
# list_all_variables.extend(list_variables_treatments)
# list_all_variables.extend(list_variables_cancer_characteristics)
# list_all_variables.extend(list_variables_demographics)
# list_all_variables.extend(list_variables_smoking)
# list_all_variables.extend(list_variables_family)
# list_all_variables.extend(list_variables_body)


all_variables_names = []

for var in list_all_variables:
    all_variables_names.append(var[0])

print(all_variables_names)

df_variables = df_pros[all_variables_names]

# print("sum before missing data")
# print(df_variables.isnull().sum())



# for i in range(df_variables.shape[0]):
#     for j in range(df_variables.shape[1]):
#
#         if(pd.isna(df_variables.iloc[i, j])):
#             rng_value = random.choice(df_variables.iloc[:, j].dropna().values)
#             df_variables.iloc[i, j] = rng_value
        

# print(df_variables["bmi_curr"])
# print(df_variables["bmi_curr"].min())
# print(df_variables["bmi_curr"].quantile(0.75))
#
# print("sum after missing data")
# print(df_variables.isnull().sum())

df = pd.DataFrame()



for var in list_all_variables:
    if(var[1] == "C"):
        df = pd.concat([df,pd.get_dummies(df_variables[var[0]])], axis = 1)
    else:
        df = pd.concat([df, df_variables[var[0]]], axis = 1)

# print("df")
# print(df)
# print(df.shape)


df = pd.concat([df, df_dummies_mortality_exitstat[["delta","xi"]]], axis = 1)

df = pd.concat([df, df_pros["mortality_exitdays"]], axis = 1)

df = df.dropna(axis = 0)

df.to_csv("pros_preprocessed_numeric_center_" + str(center) + ".csv")

