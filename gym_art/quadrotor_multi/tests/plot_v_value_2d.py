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
    # normalized_obs_dict['obs'][0][18]=2.0 - id_i

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

    normalized_obs_dict['obs'][0][18]=2.32891
    normalized_obs_dict['obs'][0][19]=0.44242
    normalized_obs_dict['obs'][0][20]=0.53575

    normalized_obs_dict['obs'][0][21]=0.63047
    normalized_obs_dict['obs'][0][22]=0.41037
    normalized_obs_dict['obs'][0][23]=0.50742

    normalized_obs_dict['obs'][0][24]=0.60510
    normalized_obs_dict['obs'][0][25]=0.39143
    normalized_obs_dict['obs'][0][26]=0.49080

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

x = np.array(
    [-1.0, -0.9500000000000001, -0.9, -0.8500000000000001, -0.8, -0.75, -0.7000000000000001, -0.65, -0.6000000000000001,
     -0.55, -0.5, -0.45, -0.4, -0.35000000000000003, -0.30000000000000004, -0.25, -0.2, -0.15000000000000002, -0.1,
     -0.05, 0.0, 0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.30000000000000004, 0.35000000000000003, 0.4, 0.45, 0.5,
     0.55, 0.6000000000000001, 0.65, 0.7000000000000001, 0.75, 0.8, 0.8500000000000001, 0.9, 0.9500000000000001, 1.0]
    )
y = np.array([-1.1800048351287842, -1.1524540185928345, -1.118280053138733, -1.0767091512680054, -1.0273784399032593, -0.9705186486244202, -0.9066543579101562, -0.8356855511665344, -0.7559000253677368, -0.6637867093086243, -0.555094838142395, -0.42685365676879883, -0.2793560028076172, -0.11700709164142609, 0.05158655345439911, 0.21327875554561615, 0.3519277572631836, 0.45398610830307007, 0.5161322951316833, 0.5477317571640015, 0.5617009997367859, 0.5667853355407715, 0.5672686696052551, 0.5649275779724121, 0.5604110360145569, 0.5539120435714722, 0.5454598665237427, 0.5350514650344849, 0.522705078125, 0.5084913969039917, 0.492544949054718, 0.47505366802215576, 0.45622169971466064, 0.4362058639526367, 0.4150715470314026, 0.3928239345550537, 0.3695249557495117, 0.34542858600616455, 0.32100898027420044, 0.2968332767486572, 0.27337342500686646]
             )

xmax = x[np.argmax(y)]
ymax = y.max()
text = "max value={:.5f}, x={:.2f}".format(ymax, xmax)

fig = px.scatter(x=x, y=y, title=text)
fig.show()
