import numpy as np

"""
# How to use:
1. Go to Sample-Factory, actor_critic.py: ActorCriticSeparateWeights: forward function
2. Set a debug point at the first line of the forward function.
    x = self.forward_head(normalized_obs_dict)
3. In PyCharm, go to Console, copy and past code below

tmp_score=[]
idx = []
for i in range(-20, 21): 
    id_i = i * 0.05
    normalized_obs_dict['obs'][0][2]=id_i
    normalized_obs_dict['obs'][0][18]=2.0 - id_i

    normalized_obs_dict['obs'][0][0]=0.11225
    normalized_obs_dict['obs'][0][1]=-0.10613

    
    # vel = 0
    normalized_obs_dict['obs'][0][3]=0.0
    normalized_obs_dict['obs'][0][4]=0.0
    normalized_obs_dict['obs'][0][5]=0.0
    
    normalized_obs_dict['obs'][0][6]=-1.0
    normalized_obs_dict['obs'][0][7]=0.0
    normalized_obs_dict['obs'][0][8]=0.0
    
    normalized_obs_dict['obs'][0][9]=0.0
    normalized_obs_dict['obs'][0][10]=-1.0
    normalized_obs_dict['obs'][0][11]=0.0
    
    normalized_obs_dict['obs'][0][12]=0.0
    normalized_obs_dict['obs'][0][13]=0.0
    normalized_obs_dict['obs'][0][14]=1.0
    
    normalized_obs_dict['obs'][0][15]=0.0
    normalized_obs_dict['obs'][0][16]=0.0
    normalized_obs_dict['obs'][0][17]=0.0
    
    normalized_obs_dict['obs'][0][19]=2.32891
    normalized_obs_dict['obs'][0][20]=0.44242
    normalized_obs_dict['obs'][0][21]=0.53575

    normalized_obs_dict['obs'][0][22]=0.63047
    normalized_obs_dict['obs'][0][23]=0.41037
    normalized_obs_dict['obs'][0][24]=0.50742

    normalized_obs_dict['obs'][0][25]=0.60510
    normalized_obs_dict['obs'][0][26]=0.39143
    normalized_obs_dict['obs'][0][27]=0.49080

    # normalized_obs_dict['obs'][0][19]=1.96745
    # normalized_obs_dict['obs'][0][20]=1.89243
    # normalized_obs_dict['obs'][0][21]=1.81948
    # 
    # normalized_obs_dict['obs'][0][22]=2.03370
    # normalized_obs_dict['obs'][0][23]=1.96089
    # normalized_obs_dict['obs'][0][24]=1.89022
    # 
    # normalized_obs_dict['obs'][0][25]=2.10229
    # normalized_obs_dict['obs'][0][26]=2.03162
    # normalized_obs_dict['obs'][0][27]=1.96316
    
    
    
    x = self.forward_head(normalized_obs_dict)
    x, new_rnn_states = self.forward_core(x, rnn_states)
    result = self.forward_tail(x, values_only, sample_actions=True)
    tmp_score.append(result['values'].item())
    idx.append(id_i)

print(tmp_score)
print(idx)

4. Copy and paste the print info and replace v_value dict below. 
"""


import plotly.express as px

x = np.array([-1.0, -0.9500000000000001, -0.9, -0.8500000000000001, -0.8, -0.75, -0.7000000000000001, -0.65, -0.6000000000000001, -0.55, -0.5, -0.45, -0.4, -0.35000000000000003, -0.30000000000000004, -0.25, -0.2, -0.15000000000000002, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.30000000000000004, 0.35000000000000003, 0.4, 0.45, 0.5, 0.55, 0.6000000000000001, 0.65, 0.7000000000000001, 0.75, 0.8, 0.8500000000000001, 0.9, 0.9500000000000001, 1.0]
)
y = np.array([-0.9369750022888184, -0.9423980116844177, -0.9493765234947205, -0.9578582048416138, -0.9677219390869141, -0.9787406921386719, -0.9905290007591248, -1.0024943351745605, -1.0138028860092163, -1.0233829021453857, -1.0299676656723022, -1.0321635007858276, -1.0285038948059082, -1.0174673795700073, -0.9974603056907654, -0.9668028354644775, -0.9237292408943176, -0.8662384748458862, -0.7912389039993286, -0.6923562288284302, -0.5586671233177185, -0.38552260398864746, -0.20245279371738434, -0.05835279822349548, 0.04142849147319794, 0.11664189398288727, 0.17627333104610443, 0.21909382939338684, 0.24408745765686035, 0.25409162044525146, 0.2540130615234375, 0.24879544973373413, 0.24222531914710999, 0.236468106508255, 0.23230469226837158, 0.2296997308731079, 0.22828742861747742, 0.22764521837234497, 0.22740855813026428, 0.22730135917663574, 0.22713640332221985]
)

xmax = x[np.argmax(y)]
ymax = y.max()
text = "max value={:.5f}, x={:.2f}".format(ymax, xmax)

fig = px.scatter(x=x, y=y, title=text)
fig.show()