import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, joblib, time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, classification_report)
from sklearn.inspection import permutation_importance
sns.set_theme(style='whitegrid'); RS=42
import os; os.makedirs('figures',exist_ok=True)

df = pd.read_csv('cardio_train.csv', sep=';').drop(columns='id')
print(df.shape, '| missing:', df.isna().sum().sum(), '| duplicates:', df.duplicated().sum())
print(df.head()); display(df.describe().T)
print(df.cardio.value_counts(normalize=True).round(3))

d = df.copy()
n0 = len(d)
d = d.drop_duplicates()
d['age_years'] = (d.age/365.25).round(1)
# physiologically implausible values
mask = (d.ap_hi.between(70,250)) & (d.ap_lo.between(40,200)) & (d.ap_hi > d.ap_lo) \
     & (d.height.between(130,220)) & (d.weight.between(35,200))
print(f'Removed duplicates + implausible rows: {n0-mask.sum()} of {n0} ({(n0-mask.sum())/n0:.2%})')
d = d[mask].reset_index(drop=True); print('Clean shape:', d.shape)
fig,ax=plt.subplots(1,4,figsize=(16,3.2))
for a,c in zip(ax,['age_years','ap_hi','ap_lo','weight']): sns.boxplot(x=d[c],ax=a,color='steelblue'); a.set_title(c)
plt.tight_layout(); plt.savefig('figures/01_boxplots_clean.png',dpi=130); plt.show()

d['bmi'] = d.weight/(d.height/100)**2
d['pulse_pressure'] = d.ap_hi - d.ap_lo
d['gender'] = (d.gender==2).astype(int)           # 1 = male
d = d.drop(columns=['age'])
d = d[d.bmi.between(12,60)].reset_index(drop=True)
# One-hot for nominal-like ordinal risk levels (used by distance/probability models); trees handle either way
d = pd.get_dummies(d, columns=['cholesterol','gluc'], prefix=['chol','gluc'], dtype=int)
X, y = d.drop(columns='cardio'), d.cardio
print(X.shape); print(list(X.columns))

fig,ax=plt.subplots(2,3,figsize=(16,8))
for a,c in zip(ax.flat,['age_years','ap_hi','bmi','weight','pulse_pressure','ap_lo']):
    sns.kdeplot(data=d,x=c,hue='cardio',fill=True,common_norm=False,ax=a,palette=['#2a9d8f','#e63946']); a.set_title(c)
plt.tight_layout(); plt.savefig('figures/02_kde_risk_factors.png',dpi=130); plt.show()

fig,ax=plt.subplots(1,2,figsize=(16,6))
corr=pd.concat([X,y],axis=1).corr()
sns.heatmap(corr,cmap='coolwarm',center=0,ax=ax[0],cbar=False,xticklabels=True,yticklabels=True); ax[0].set_title('Correlation matrix')
corr['cardio'].drop('cardio').sort_values().plot.barh(ax=ax[1],color='steelblue'); ax[1].set_title('Correlation with cardio')
plt.tight_layout(); plt.savefig('figures/03_correlation.png',dpi=130); plt.show()

print((d.groupby(pd.cut(d.age_years,[28,40,50,55,60,66]))['cardio'].mean()).round(3))

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=RS)
print(Xtr.shape, Xte.shape)
cont = ['age_years','height','weight','ap_hi','ap_lo','bmi','pulse_pressure']
def prep(): return ColumnTransformer([('s',StandardScaler(),cont)],remainder='passthrough')
cv = StratifiedKFold(5,shuffle=True,random_state=RS)

sub_idx = Xtr.sample(15000,random_state=RS).index
Xs,ys = Xtr.loc[sub_idx], ytr.loc[sub_idx]
cv3 = StratifiedKFold(3,shuffle=True,random_state=RS)

grids = {
 'Decision Tree': (Pipeline([('p',prep()),('m',DecisionTreeClassifier(random_state=RS))]),
     {'m__max_depth':[3,4,5,6,8,10],'m__min_samples_leaf':[20,50,100],'m__criterion':['gini','entropy']}, (Xtr,ytr), cv),
 'Naive Bayes': (Pipeline([('p',prep()),('m',GaussianNB())]),
     {'m__var_smoothing':[1e-9,1e-7,1e-5,1e-3,1e-1]}, (Xtr,ytr), cv),
 'KNN': (Pipeline([('p',prep()),('m',KNeighborsClassifier(n_jobs=-1))]),
     {'m__n_neighbors':[5,15,31,51,81],'m__weights':['uniform','distance']}, (Xs,ys), cv3),
 'ANN (MLP)': (Pipeline([('p',prep()),('m',MLPClassifier(early_stopping=True,max_iter=300,random_state=RS))]),
     {'m__hidden_layer_sizes':[(32,),(64,32)],'m__alpha':[1e-4,1e-2],'m__learning_rate_init':[1e-3]}, (Xs,ys), cv3),
}
best, tune_log = {}, []
for name,(pipe,grid,(Xg,yg),cvx) in grids.items():
    t=time.time(); gs=GridSearchCV(pipe,grid,scoring='f1',cv=cvx,n_jobs=1).fit(Xg,yg)
    est = gs.best_estimator_.fit(Xtr,ytr)   # refit on full training set
    best[name]=est; tune_log.append((name,gs.best_params_,round(gs.best_score_,4),round(time.time()-t,1)))
    print(f'{name:14s} best CV-F1={gs.best_score_:.4f}  {gs.best_params_}  ({time.time()-t:.0f}s)')

scoring = ['accuracy','precision','recall','f1','roc_auc']
rows=[]
for name,est in best.items():
    r = cross_validate(est,Xtr,ytr,cv=cv,scoring=scoring,n_jobs=1)
    rows.append({'Model':name, **{m.upper() if m=='roc_auc' else m.capitalize():f"{r['test_'+m].mean():.4f} ± {r['test_'+m].std():.4f}" for m in scoring}})
cv_df = pd.DataFrame(rows).set_index('Model'); display(cv_df)

res, preds, probs = [], {}, {}
for name,est in best.items():
    t=time.time(); p=est.predict(Xte); pr=est.predict_proba(Xte)[:,1]; preds[name]=p; probs[name]=pr
    res.append({'Model':name,'Accuracy':accuracy_score(yte,p),'Precision':precision_score(yte,p),
                'Recall':recall_score(yte,p),'F1':f1_score(yte,p),'ROC-AUC':roc_auc_score(yte,pr)})
res_df = pd.DataFrame(res).set_index('Model').round(4)
print(res_df.style.highlight_max(color='#b7e4c7'))
res_df.to_csv('test_metrics.csv')
res_df.plot.bar(figsize=(11,4.5),rot=0,colormap='viridis',ylim=(0.5,0.85)); plt.title('Hold-out test performance'); plt.legend(loc='lower right',ncol=5)
plt.tight_layout(); plt.savefig('figures/04_metric_comparison.png',dpi=130); plt.show()

fig,ax=plt.subplots(1,4,figsize=(20,4.5))
for a,(name,p) in zip(ax,preds.items()):
    ConfusionMatrixDisplay(confusion_matrix(yte,p),display_labels=['No CVD','CVD']).plot(ax=a,cmap='Blues',colorbar=False,values_format='d'); a.set_title(name)
plt.tight_layout(); plt.savefig('figures/05_confusion_matrices.png',dpi=130); plt.show()
for name,p in preds.items():
    tn,fp,fn,tp = confusion_matrix(yte,p).ravel()
    print(f'{name:14s} TN={tn} FP={fp} FN={fn} TP={tp} | Specificity={tn/(tn+fp):.3f} Sensitivity={tp/(tp+fn):.3f}')
plt.figure(figsize=(6.5,5.5))
for name,pr in probs.items():
    f,t,_=roc_curve(yte,pr); plt.plot(f,t,label=f'{name} (AUC={roc_auc_score(yte,pr):.3f})')
plt.plot([0,1],[0,1],'k--',alpha=.4); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC curves'); plt.legend(); plt.tight_layout()
plt.savefig('figures/06_roc.png',dpi=130); plt.show()

from scipy import stats
cvF1 = {n:cross_validate(e,Xtr,ytr,cv=cv,scoring='f1')['test_score'] for n,e in best.items()}
top = max(cvF1,key=lambda k:cvF1[k].mean()); print('Best CV model:',top)
for n,s in cvF1.items():
    if n!=top: print(f'{top} vs {n}: mean diff={cvF1[top].mean()-s.mean():+.4f}, paired t-test p={stats.ttest_rel(cvF1[top],s).pvalue:.4f}')

# (a) Decision-tree native importance
dt = best['Decision Tree']; names = cont + [c for c in X.columns if c not in cont]
imp = pd.Series(dt.named_steps['m'].feature_importances_, index=names).sort_values()
imp.tail(10).plot.barh(figsize=(7,4.5),color='#e63946'); plt.title('Decision Tree — Gini importance (top 10)'); plt.tight_layout()
plt.savefig('figures/07_dt_importance.png',dpi=130); plt.show()

# (b) Model-agnostic permutation importance (F1 drop) on 3,000 test rows
Xp,yp = Xte.sample(3000,random_state=RS), yte.loc[Xte.sample(3000,random_state=RS).index]
perm = {}
for name,est in best.items():
    r = permutation_importance(est,Xp,yp,scoring='f1',n_repeats=5,random_state=RS,n_jobs=1)
    perm[name]=pd.Series(r.importances_mean,index=X.columns)
perm_df = pd.DataFrame(perm); perm_df.loc[perm_df.mean(axis=1).sort_values(ascending=False).index].head(10).round(4)

top = perm_df.loc[perm_df.mean(axis=1).sort_values(ascending=False).head(10).index]
print(top.round(4))
top.iloc[::-1].plot.barh(figsize=(9,6)); plt.xlabel('Mean F1 drop when feature shuffled'); plt.title('Permutation importance — all models (top 10 by mean)')
plt.tight_layout(); plt.savefig('figures/08_perm_importance.png',dpi=130); plt.show()

plt.figure(figsize=(20,7)); plot_tree(dt.named_steps['m'],feature_names=names,class_names=['No CVD','CVD'],max_depth=3,filled=True,rounded=True,fontsize=8)
plt.title('Decision Tree (top 3 levels) — interpretable clinical rules'); plt.savefig('figures/09_tree.png',dpi=120,bbox_inches='tight'); plt.show()

# Decision-tree depth vs train/test F1 (over/under-fitting)
ds=range(1,21); tr,te=[],[]
for k in ds:
    m=Pipeline([('p',prep()),('m',DecisionTreeClassifier(max_depth=k,min_samples_leaf=20,random_state=RS))]).fit(Xtr,ytr)
    tr.append(f1_score(ytr,m.predict(Xtr))); te.append(f1_score(yte,m.predict(Xte)))
# KNN k vs F1 (on 15k subsample for speed, evaluated on test)
ks=[1,3,5,11,21,41,61,101]; ktr,kte=[],[]
for k in ks:
    m=Pipeline([('p',prep()),('m',KNeighborsClassifier(k,n_jobs=-1))]).fit(Xs,ys)
    ktr.append(f1_score(ys,m.predict(Xs))); kte.append(f1_score(yte,m.predict(Xte)))
fig,ax=plt.subplots(1,3,figsize=(17,4))
ax[0].plot(ds,tr,label='train'); ax[0].plot(ds,te,label='test'); ax[0].set_title('Decision Tree: depth vs F1'); ax[0].set_xlabel('max_depth'); ax[0].legend()
ax[1].plot(ks,ktr,label='train'); ax[1].plot(ks,kte,label='test'); ax[1].set_title('KNN: k vs F1'); ax[1].set_xlabel('k'); ax[1].legend()
mlp=best['ANN (MLP)'].named_steps['m']; ax[2].plot(mlp.loss_curve_,label='train loss'); ax[2].plot(1-np.array(mlp.validation_scores_),label='val error'); ax[2].set_title('ANN training curves'); ax[2].set_xlabel('epoch'); ax[2].legend()
plt.tight_layout(); plt.savefig('figures/10_behaviour.png',dpi=130); plt.show()

# Timing
tm=[]
for name,est in best.items():
    t=time.time(); est.fit(Xtr,ytr); ft=time.time()-t; t=time.time(); est.predict(Xte); pt=time.time()-t
    tm.append({'Model':name,'Train (s)':round(ft,2),'Predict 14k rows (s)':round(pt,3)})
pd.DataFrame(tm).set_index('Model')

bn = max(res_df.F1.items(), key=lambda kv:kv[1])[0]; print('Best model by test F1:',bn)
err = Xte.copy(); err['y']=yte; err['pred']=preds[bn]; err['prob']=probs[bn]
err['kind']=np.select([(err.y==1)&(err.pred==0),(err.y==0)&(err.pred==1)],['False Negative','False Positive'],'Correct')
print(err.groupby('kind')[['age_years','ap_hi','ap_lo','bmi','prob']].mean().round(2))
# agreement between models
P=pd.DataFrame(preds); print('Pairwise agreement:'); display(P.corr().round(3))

sw = pd.DataFrame({
 'Strengths':[
  'Human-readable rules; no scaling needed; handles mixed types; fast inference; native importance',
  'Very fast; few parameters; probabilistic output; works with little data; robust baseline',
  'No training phase; non-parametric; captures local patterns; simple to explain by similar patients',
  'Models non-linear interactions; strong generalisation; probabilistic output; scales to large data'],
 'Weaknesses':[
  'High variance / overfits when deep; axis-aligned splits; unstable to small data changes',
  'Assumes conditional independence & Gaussian features — violated by correlated BP/BMI/weight',
  'Slow prediction (stores all data); sensitive to scaling, irrelevant features & choice of k',
  'Black box; many hyper-parameters; needs scaling; longer training; less clinical trust'],
 'Healthcare fit':[
  'Best for explainable triage rules & clinician buy-in',
  'Good lightweight screening baseline',
  'Useful for case-based reasoning; poor for real-time at scale',
  'Best when accuracy matters and an explainability layer (SHAP) is added']},
 index=['Decision Tree','Naive Bayes','KNN','ANN (MLP)'])
pd.set_option('display.max_colwidth',200); display(sw)

final_name = res_df.F1.idxmax(); final = best[final_name]; print('Deployed model:',final_name)
joblib.dump({'model':final,'features':list(X.columns),'name':final_name},'cvd_model.joblib')

def predict_patient(age_years,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active,model=final):
    r = dict(age_years=age_years,gender=int(gender=='male'),height=height,weight=weight,ap_hi=ap_hi,ap_lo=ap_lo,
             smoke=smoke,alco=alco,active=active,bmi=weight/(height/100)**2,pulse_pressure=ap_hi-ap_lo)
    for k in (1,2,3): r[f'chol_{k}']=int(cholesterol==k); r[f'gluc_{k}']=int(gluc==k)
    x = pd.DataFrame([r])[X.columns]; p = model.predict_proba(x)[0,1]
    band = 'LOW' if p<0.35 else 'MODERATE' if p<0.60 else 'HIGH'
    return {'risk_probability':round(float(p),3),'risk_band':band,
            'advice':{'LOW':'Routine annual check-up','MODERATE':'Lifestyle review + follow-up BP/lipid test','HIGH':'Refer for clinical cardiology assessment'}[band]}
print(predict_patient(58,'male',170,92,150,95,3,1,1,0,0))
print(predict_patient(35,'female',165,58,115,75,1,1,0,0,1))

import matplotlib.patches as mp
fig,ax=plt.subplots(figsize=(15,4.2)); ax.axis('off')
steps=[('Patient data\n(EHR / clinic form)','#a8dadc'),('Validation &\ncleaning rules','#bde0fe'),('Feature engineering\n+ scaling',"#cdb4db"),
       (f'{final_name}\nclassifier','#ffafcc'),('Risk score +\nband (Low/Mod/High)','#ffd6a5'),('Clinician dashboard\n+ explanation','#caffbf'),('Feedback &\nretraining loop','#fdffb6')]
for i,(t,c) in enumerate(steps):
    ax.add_patch(mp.FancyBboxPatch((i*2.1,0.3),1.8,1.4,boxstyle='round,pad=0.05',fc=c,ec='k')); ax.text(i*2.1+0.9,1.0,t,ha='center',va='center',fontsize=9)
    if i<len(steps)-1: ax.annotate('',xy=((i+1)*2.1-0.02,1.0),xytext=(i*2.1+1.86,1.0),arrowprops=dict(arrowstyle='->'))
ax.set_xlim(-0.2,14.8); ax.set_ylim(0,2); plt.title('Conceptual architecture: AI cardiovascular risk screening system'); plt.tight_layout()
plt.savefig('figures/11_system_architecture.png',dpi=140); plt.show()

print('FINAL TEST RESULTS'); display(res_df)
b=res_df.F1.idxmax()
print(f'Best model by F1: {b}  (F1={res_df.F1.max():.4f}, Acc={res_df.loc[b,"Accuracy"]:.4f}, AUC={res_df.loc[b,"ROC-AUC"]:.4f})')
print('Top drivers (mean permutation importance):', ', '.join(perm_df.mean(axis=1).sort_values(ascending=False).head(5).index))
