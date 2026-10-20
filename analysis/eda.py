"""Dataset decoding, integrity checks, EDA and handcrafted feature analysis."""
import argparse,csv,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,UnidentifiedImageError
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

CLASSES=["ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM"]
def inspect(root,max_per_class,seed):
    rng=np.random.default_rng(seed); rows=[]; bad=[]; hashes={}
    for split in ("NCT-CRC-HE-100K","CRC-VAL-HE-7K"):
        base=Path(root)/split
        if not base.exists(): raise FileNotFoundError(base)
        for cls in CLASSES:
            folder=base/cls
            if not folder.exists(): raise FileNotFoundError(folder)
            paths=sorted(p for p in folder.rglob("*") if p.is_file())
            for path in rng.choice(paths,size=min(len(paths),max_per_class),replace=False):
                try:
                    # Decode actual pixel content rather than trusting filename extensions
                    with Image.open(path) as img:
                        img.load(); rgb=np.asarray(img.convert("RGB"),dtype=np.uint8)
                    if rgb.ndim!=3 or rgb.shape[-1]!=3: raise ValueError("invalid RGB dimensions")
                    h=hashlib.sha256(rgb.tobytes()).hexdigest()
                    if h in hashes and hashes[h][0]!=split:
                        bad.append({"file":str(path),"issue":"identical decoded pixels across splits","matches":hashes[h][1]})
                    hashes[h]=(split,str(path))
                    flat=rgb.reshape(-1,3).astype(np.float32)/255.
                    means=flat.mean(0);stds=flat.std(0)
                    gray=np.mean(flat,axis=1)
                    rows.append(dict(split=split,category=cls,file=str(path),width=rgb.shape[1],height=rgb.shape[0],
                        r_mean=float(means[0]),g_mean=float(means[1]),b_mean=float(means[2]),
                        r_std=float(stds[0]),g_std=float(stds[1]),b_std=float(stds[2]),
                        intensity_mean=float(gray.mean()),intensity_std=float(gray.std()),
                        tissue_fraction=float((gray<.85).mean())))
                except (UnidentifiedImageError,OSError,ValueError) as exc:
                    bad.append({"file":str(path),"issue":str(exc)})
    return rows,bad

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data");p.add_argument("--out",default="results/eda")
    p.add_argument("--max-per-class",type=int,default=100);p.add_argument("--seed",type=int,default=42)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows,bad=inspect(a.data,a.max_per_class,a.seed)
    if not rows: raise RuntimeError("No images decoded")
    with (out/"image_features.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (out/"quality_issues.json").write_text(json.dumps(bad,indent=2))
    counts=Counter((r["split"],r["category"]) for r in rows)
    fig,ax=plt.subplots(figsize=(11,4))
    for i,split in enumerate(("NCT-CRC-HE-100K","CRC-VAL-HE-7K")):
        ax.bar(np.arange(9)+i*.4,[counts[(split,c)] for c in CLASSES],width=.4,label=split)
    ax.set_xticks(np.arange(9)+.2,CLASSES);ax.set_ylabel("Sampled decoded images")
    ax.legend();fig.tight_layout();fig.savefig(out/"sample_class_distribution.png");plt.close(fig)
    names=["r_mean","g_mean","b_mean","r_std","g_std","b_std","intensity_mean","intensity_std","tissue_fraction"]
    X=np.array([[r[k] for k in names] for r in rows])
    Z=PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    fig,ax=plt.subplots(figsize=(8,6))
    for cls in CLASSES:
        mask=np.array([r["category"]==cls for r in rows])
        ax.scatter(Z[mask,0],Z[mask,1],s=8,alpha=.55,label=cls)
    ax.legend(ncol=3);ax.set_title("PCA of handcrafted color/texture summary features")
    fig.tight_layout();fig.savefig(out/"feature_pca.png");plt.close(fig)
    summary={"sampled_images":len(rows),"issues":len(bad),"counts_by_split_class":{f"{s}/{c}":counts[(s,c)] for s in ("NCT-CRC-HE-100K","CRC-VAL-HE-7K") for c in CLASSES},
             "note":"All counts and integrity checks apply only to the sampled images, not the entire dataset. Duplicate checks use exact decoded pixels only."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
