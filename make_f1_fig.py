import json,glob,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='results/'; DS=['PlantVillageTomato','DeepWeeds','IP102']; LAB=['PlantVillage-Tomato','DeepWeeds','IP102']
def get(m,ds,fr):
    if fr==100: v=[json.load(open(f))['test']['macro_f1'] for f in sorted(glob.glob(f'{R}final_protocol/{m}_{ds}_seed*.json'))]
    else: v=[json.load(open(f))['test_macro_f1'] for f in sorted(glob.glob(f'{R}extra_experiments/{m}_{ds}_frac{fr}_seed*.json'))]
    v=np.array(v,float); 
    if v.max()<=1.0: v*=100
    return v.mean(),v.std()
fig,ax=plt.subplots(1,3,figsize=(10,3.3),sharey=False)
cols={'M2':'#9a9a9a','M1':'#1f5fa8'}
for a,fr in zip(ax,[10,25,100]):
    for k,m in enumerate(['M2','M1']):
        mu=[get(m,d,fr)[0] for d in DS]; sd=[get(m,d,fr)[1] for d in DS]
        a.bar(np.arange(3)+(k-.5)*.36,mu,.36,yerr=sd,capsize=2,color=cols[m],label='M2 (CE)' if m=='M2' else 'M1 (composite)')
    a.set_xticks(range(3)); a.set_xticklabels(['PV-Tomato','DeepWeeds','IP102'],fontsize=8)
    a.set_title(f'{fr} % of training data',fontsize=9); a.set_ylim(0,105)
    a.spines[['top','right']].set_visible(False)
ax[0].set_ylabel('Test macro-F1 (%)'); ax[0].legend(fontsize=8,frameon=False,loc='lower left')
plt.tight_layout(); plt.savefig('figures_final5/fig10_macro_f1.png',dpi=300)
for fr in [10,25,100]: print(fr,[tuple(round(x,2) for x in get(m,d,fr)) for d in DS for m in ['M2','M1']])
