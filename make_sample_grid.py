import os,random,sys
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from agroonto.dataset import build_loaders
random.seed(7)
O="figures_final4"; os.makedirs(O,exist_ok=True)
DS=[("PlantVillageTomato","PlantVillage-Tomato"),("DeepWeeds","DeepWeeds"),("IP102","IP102")]
NCLS,NIMG=4,3   # classes per dataset, images per class
ident=lambda x:x
fig,axes=plt.subplots(len(DS),NCLS*NIMG,figsize=(NCLS*NIMG*1.25,len(DS)*1.55+0.6))
for r,(d,title) in enumerate(DS):
    tl,_,names,K=build_loaders(d,ident)
    smp=tl.dataset.samples
    by={}
    for p,l in smp: by.setdefault(l,[]).append(p)
    cls=sorted(random.sample([c for c in by if len(by[c])>=NIMG],NCLS))
    for ci,c in enumerate(cls):
        for ii,p in enumerate(random.sample(by[c],NIMG)):
            ax=axes[r,ci*NIMG+ii]; ax.imshow(Image.open(p).convert("RGB").resize((224,224))); ax.axis("off")
            if ii==1: ax.set_title(" ".join(str(names[c]).replace("_"," ").replace("Tomato","").split()).strip().replace("pest ","Pest class ")[:34],fontsize=7,pad=3)
    axes[r,0].text(-0.08,0.5,title,transform=axes[r,0].transAxes,rotation=90,va="center",ha="right",fontsize=9,fontweight="bold")
plt.tight_layout(); plt.savefig(f"{O}/fig0_sample_grid_v2.png",dpi=300,bbox_inches="tight"); print("saved")
