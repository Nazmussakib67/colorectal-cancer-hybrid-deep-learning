"""CRC histopathology: CNN, classical, and quantum baselines. Research only."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml
import joblib

CLASSES={"ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM"}

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load_data(root):
    root=Path(root)
    norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    base=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),norm])
    aug=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ToTensor(),norm])
    tr=root/"NCT-CRC-HE-100K"; ex=root/"CRC-VAL-HE-7K"
    for p in (tr,ex):
        if not p.exists() or {x.name for x in p.iterdir() if x.is_dir()}!=CLASSES:
            raise ValueError(f"Expected nine class subdirectories in {p}")
    a=datasets.ImageFolder(tr,transform=aug); b=datasets.ImageFolder(tr,transform=base)
    c=datasets.ImageFolder(ex,transform=base)
    assert b.classes==c.classes
    return a,b,c

def labels(ds):
    return np.array([int(y==ds.class_to_idx["TUM"]) for _,y in ds.samples])

def dl(ds,indices,batch,shuffle=False):
    return DataLoader(Subset(ds,indices),batch_size=batch,shuffle=shuffle,num_workers=2,pin_memory=torch.cuda.is_available())

def model_init():
    m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.fc=nn.Linear(m.fc.in_features,2)
    return m

def train(m,tr,va,device,epochs,lr,out):
    opt=torch.optim.AdamW(m.parameters(),lr=lr)
    lossfn=nn.CrossEntropyLoss(); best=float("inf")
    for ep in range(epochs):
        m.train(); total=0
        for x,y in tr:
            x=x.to(device); y=(y==tr.dataset.dataset.class_to_idx["TUM"]).long().to(device)
            opt.zero_grad(); loss=lossfn(m(x),y); loss.backward(); opt.step(); total+=loss.item()
        m.eval(); losses=[]
        with torch.no_grad():
            for x,y in va:
                y=(y==va.dataset.dataset.class_to_idx["TUM"]).long().to(device)
                losses.append(lossfn(m(x.to(device)),y).item())
        v=np.mean(losses); print(f"Epoch {ep+1}: train={total/len(tr):.4f}, validation={v:.4f}")
        if v<best:
            best=v; torch.save(m.state_dict(),out/"cnn_best.pt")
    m.load_state_dict(torch.load(out/"cnn_best.pt",map_location=device,weights_only=True))

def features(m,loader,device):
    m.eval(); feats=[]; ys=[]; probs=[]; captured=[]
    hook=m.fc.register_forward_hook(lambda mod,inp,output:captured.append(inp[0].detach()))
    with torch.no_grad():
        for x,y in loader:
            captured.clear(); pred=m(x.to(device))
            feats.append(captured[0].cpu().numpy()); ys.append((y==loader.dataset.dataset.class_to_idx["TUM"]).numpy().astype(int))
            probs.append(pred.softmax(1)[:,1].cpu().numpy())
    hook.remove()
    return np.concatenate(feats),np.concatenate(ys),np.concatenate(probs)

def vqc_model(layers):
    dev=qml.device("default.qubit",wires=4)
    @qml.qnode(dev,interface="torch",diff_method="backprop")
    def circuit(x,w):
        qml.AngleEmbedding(x,wires=range(4),rotation="Y")
        qml.StronglyEntanglingLayers(w,wires=range(4))
        return qml.expval(qml.PauliZ(0))
    class VQC(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights=nn.Parameter(torch.randn(layers,4,3)*.05)
            self.scale=nn.Parameter(torch.tensor(1.))
            self.bias=nn.Parameter(torch.tensor(0.))
        def forward(self,x):
            return self.scale*torch.stack([circuit(row,self.weights) for row in x])+self.bias
    return VQC()

def fit_vqc(x,y,epochs,layers,out):
    net=vqc_model(layers); opt=torch.optim.Adam(net.parameters(),lr=.02)
    xx=torch.tensor(x,dtype=torch.float32); yy=torch.tensor(y,dtype=torch.float32)
    for ep in range(epochs):
        order=torch.randperm(len(xx)); losses=[]
        for ids in order.split(16):
            opt.zero_grad()
            loss=nn.functional.binary_cross_entropy_with_logits(net(xx[ids]),yy[ids])
            loss.backward();opt.step();losses.append(loss.item())
        print(f"VQC epoch {ep+1}: {np.mean(losses):.4f}")
    torch.save(net.state_dict(),out/"vqc.pt")
    return net

def predict_vqc(net,x):
    net.eval(); arr=[]
    with torch.no_grad():
        for i in range(0,len(x),32):
            arr.append(torch.sigmoid(net(torch.tensor(x[i:i+32],dtype=torch.float32))).numpy())
    return np.concatenate(arr)

def evaluate(y,p):
    pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),specificity=float(tn/(tn+fp)) if tn+fp else None,f1=float(f1_score(y,pred,zero_division=0)),roc_auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]])

def plots(y,probs,out):
    fig,ax=plt.subplots()
    for name,p in probs.items():
        fpr,tpr,_=roc_curve(y,p)
        ax.plot(fpr,tpr,label=f"{name}: AUC={roc_auc_score(y,p):.3f}")
        cm=confusion_matrix(y,p>=.5,labels=[0,1])
        f,a=plt.subplots();a.imshow(cm,cmap="Blues")
        a.set(xlabel="Predicted",ylabel="Actual",title=name)
        for i in range(2):
            for j in range(2): a.text(j,i,str(cm[i,j]),ha="center",va="center")
        f.savefig(out/f"confusion_{name}.png");plt.close(f)
    ax.plot([0,1],[0,1],"--",color="gray")
    ax.set(xlabel="False positive rate",ylabel="True positive rate",title="Independent validation")
    ax.legend();fig.savefig(out/"roc.png");plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data");p.add_argument("--output",default="results")
    p.add_argument("--epochs",type=int,default=8);p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--quantum-epochs",type=int,default=10);p.add_argument("--quantum-layers",type=int,default=2)
    p.add_argument("--quantum-train-limit",type=int,default=2000)
    p.add_argument("--max-train",type=int,default=0);p.add_argument("--max-external",type=int,default=0)
    args=p.parse_args();seed_all(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/"config.json").write_text(json.dumps(vars(args),indent=2))
    a,b,c=load_data(args.data)
    ids=np.arange(len(a)); yy=labels(a)
    tr,va=train_test_split(ids,test_size=.2,random_state=args.seed,stratify=yy)
    if args.max_train and args.max_train<len(tr):
        tr,_=train_test_split(tr,train_size=args.max_train,random_state=args.seed,stratify=yy[tr])
    ex=np.arange(len(c)); cy=labels(c)
    if args.max_external and args.max_external<len(ex):
        ex,_=train_test_split(ex,train_size=args.max_external,random_state=args.seed,stratify=cy)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    m=model_init().to(device)
    train(m,dl(a,tr,args.batch_size,True),dl(b,va,args.batch_size),device,args.epochs,args.lr,out)
    train_x,train_y,_=features(m,dl(b,tr,args.batch_size),device)
    ext_x,ext_y,cnn_p=features(m,dl(c,ex,args.batch_size),device)
    rng=np.random.default_rng(args.seed)
    chosen,_=train_test_split(np.arange(len(train_y)),train_size=min(args.quantum_train_limit,len(train_y)-1),random_state=args.seed,stratify=train_y)
    scaler=StandardScaler().fit(train_x[chosen])
    pca=PCA(n_components=4,random_state=args.seed).fit(scaler.transform(train_x[chosen]))
    xt=np.clip(pca.transform(scaler.transform(train_x[chosen])),-np.pi,np.pi)
    xe=np.clip(pca.transform(scaler.transform(ext_x)),-np.pi,np.pi)
    joblib.dump(scaler,out/"scaler.joblib");joblib.dump(pca,out/"pca.joblib")
    classical=LogisticRegression(max_iter=2000,random_state=args.seed).fit(xt,train_y[chosen])
    joblib.dump(classical,out/"logreg.joblib")
    quantum=fit_vqc(xt,train_y[chosen],args.quantum_epochs,args.quantum_layers,out)
    probs={"CNN":cnn_p,"PCA_LogReg":classical.predict_proba(xe)[:,1],"PCA_VQC":predict_vqc(quantum,xe)}
    report={k:evaluate(ext_y,v) for k,v in probs.items()}
    (out/"metrics.json").write_text(json.dumps(report,indent=2))
    plots(ext_y,probs,out);print(json.dumps(report,indent=2))

if __name__=="__main__":main()
