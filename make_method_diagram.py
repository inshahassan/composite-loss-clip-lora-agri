import glob,random
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image
random.seed(3)
img=Image.open(random.choice(sorted(glob.glob("data/plantvillage/PlantVillage/train/Tomato___Early_blight/*")))).convert("RGB").resize((224,224))
fig,ax=plt.subplots(figsize=(13,5.2)); ax.set_xlim(0,13); ax.set_ylim(0,5.5); ax.axis("off")
def box(x,y,w,h,t,c,fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03,rounding_size=0.08",fc=c,ec="#333",lw=1.2)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs)
def arr(p,q,ls="-",rad=0.0,col="#333"): ax.add_patch(FancyArrowPatch(p,q,arrowstyle="->",mutation_scale=14,lw=1.4,color=col,linestyle=ls,connectionstyle=f"arc3,rad={rad}"))
ax.imshow(img,extent=(0.25,1.65,3.3,4.7),zorder=3); ax.add_patch(plt.Rectangle((0.25,3.3),1.4,1.4,fill=False,ec="#333",lw=1.2,zorder=4)); ax.text(0.95,3.05,"Input image\n224x224",ha="center",va="center",fontsize=9)
box(2.2,2.9,3.3,1.9,"CLIP ViT-B/32 image encoder\nblocks 0-7 frozen\nblocks 8-11 + LoRA (r=8, alpha=16)\n(trainable)","#cfe4f5")
box(2.2,0.35,3.3,1.2,"Precomputed CLIP text embeddings\n(frozen) + learnable class delta","#e6e6e6")
box(6.0,3.3,1.4,1.1,"Image\nfeature","#f2f2f2"); box(6.0,0.4,1.4,1.1,"Class text\nfeatures","#f2f2f2")
box(7.9,3.9,1.5,0.9,"Linear head\nlogits","#fde7b8"); box(7.9,2.2,1.5,0.9,"CLIP-style logits\n(cosine x scale)","#fde7b8")
ax.add_patch(Circle((10.0,3.1),0.22,fc="#fff",ec="#333",lw=1.2)); ax.text(10.0,3.1,"avg",ha="center",va="center",fontsize=7)
box(10.7,4.05,2.0,0.95,"Cross-entropy\n(hard labels)","#e1f0d9"); box(10.7,2.6,2.0,0.95,"Soft-target KL\n(uniform + S)","#f8d9d9"); box(10.7,1.1,2.0,0.95,"Prototype loss\n(MSE to S)","#f8d9d9")
box(7.9,0.15,1.9,0.8,"Taxonomy similarity\nmatrix S (train only)","#fff5cc")
arr((1.65,4.0),(2.2,3.9)); arr((5.5,3.85),(6.0,3.85)); arr((5.5,0.95),(6.0,0.95))
arr((7.4,4.0),(7.9,4.3)); arr((7.4,3.5),(7.9,2.75)); arr((7.4,1.1),(7.9,2.35)); 
arr((9.4,4.35),(10.7,4.5)); arr((9.4,4.1),(9.85,3.3)); arr((9.4,2.65),(9.85,2.9)); arr((10.22,3.1),(10.7,3.1))
arr((6.7,3.3),(10.7,1.6),rad=0.42)
arr((9.8,0.55),(10.7,1.25),ls="--"); arr((9.0,0.95),(10.9,2.6),ls="--",rad=-0.15)
ax.text(6.5,5.38,"Trainable: LoRA, class delta, linear head, logit scale (~0.4M parameters). S enters only the training loss (red). Inference: image encoder + linear head.",ha="center",fontsize=8.3,style="italic")
plt.savefig("figures_final5/fig9_method_overview_v5.png",dpi=300,bbox_inches="tight")
