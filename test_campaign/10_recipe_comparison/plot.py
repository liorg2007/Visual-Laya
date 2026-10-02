import json,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r=json.load(open('results.json'))
I=r['full_image']['groups']['image_macro(3 tasks)']; T=r['full_text']['groups']['text_retention(ag_news,boolq)']
pts={n:(T['point'][n],I['point'][n],T['ci'][n],I['ci'][n]) for n in I['point'] if n in T['point']}
pts['stock_laya']=(T['point']['stock_laya'],None,T['ci']['stock_laya'],None)
fig,ax=plt.subplots(figsize=(6,4.5))
for n,(x,y,cx,cy) in pts.items():
    if y is None: ax.axvline(x,color='gray',ls='--'); ax.text(x,0.45,'stock Laya (text)',rotation=90,va='bottom',ha='right',fontsize=8); continue
    ax.errorbar(x,y,xerr=[[x-cx[0]],[cx[1]-x]],yerr=[[y-cy[0]],[cy[1]-y]],fmt='o',capsize=2); ax.annotate(n,(x,y),textcoords='offset points',xytext=(5,5),fontsize=8)
ax.set_xlabel('text retention: mean acc (AG News, BoolQ)'); ax.set_ylabel('image macro acc (EuroSAT, Pets, VQAv2 y/n)')
ax.set_title('Accuracy vs text retention (full held-out test sets, 95% cluster bootstrap)',fontsize=9)
plt.tight_layout(); plt.savefig('tradeoff.png',dpi=130)
