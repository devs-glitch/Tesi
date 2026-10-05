import json, glob, collections, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
  'axes.edgecolor':'#8a8984','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','axes.linewidth':0.6})
C=['Blip','Extremely_Loud','Scattered_Light','Violin_Mode']
arms=[('weight-only','Weights only','#2a78d6','o'),('membrane-only','Membrane only','#eb6834','s'),('joint','Joint','#1baf7a','^')]
bits=[32,8,4,2]; X=np.arange(4)
acc=collections.defaultdict(dict); iou=collections.defaultdict(dict); en=collections.defaultdict(dict)
g=json.load(open('results/grid_summary.json'))['results']
for r in g.values():
    if r['arm']=='fp32-control': continue
    acc[r['arm']].setdefault(r['bits'],[]).append(r['qat_accuracy'])
d=json.load(open('results/divergence.json'))['runs']
for r in d.values():
    if r['arm']=='fp32-control': continue
    b=r['weight_bits'] or r['membrane_bits']
    iou[r['arm']].setdefault(b,[]).append(np.mean([r['divergence'][c]['iou'] for c in C]))
macs=json.load(open('results/energy.json'))['macs']; T=8
pj={None:(4.6,0.9),8:(0.2,0.03),4:(0.05,0.015),2:(0.0125,0.0075)}
def E(rt,wb):
    m,a=pj[wb]; return macs['conv1']*m + a*T*(rt['1']*macs['conv2']+rt['2']*macs['conv3']+rt['3']*macs['conv4']+rt['4']*macs['linear'])
ref={}
for f in glob.glob('results/grid/fp32_112px_seed????.json'):
    j=json.load(open(f)); ref[j['seed']]=E(j['mean_rate'],None)
for f in glob.glob('results/grid/qat_*_112px_seed????.json'):
    j=json.load(open(f))
    if j['arm']=='fp32-control': continue
    b=j['weight_bits'] or j['membrane_bits']
    en[j['arm']].setdefault(b,[]).append(E(j['mean_rate'],j['weight_bits'])/ref[j['seed']])
fp32acc=[0.9833,0.9819,0.9847]
fig,ax=plt.subplots(1,3,figsize=(7.2,2.6),constrained_layout=True)
off={'weight-only':-0.12,'membrane-only':0,'joint':0.12}
for arm,lab,col,mk in arms:
    for k,(store,base) in enumerate([(acc,fp32acc),(iou,[1,1,1]),(en,[1,1,1])]):
        ys=[np.mean(base)]+[np.mean(store[arm][b]) for b in (8,4,2)]
        lo=[np.min(base)]+[np.min(store[arm][b]) for b in (8,4,2)]
        hi=[np.max(base)]+[np.max(store[arm][b]) for b in (8,4,2)]
        x=X+off[arm]
        ax[k].plot(x,ys,color=col,lw=2,marker=mk,ms=6,mec='#fcfcfb',mew=1.2,label=lab,zorder=3)
        ax[k].vlines(x,lo,hi,color=col,lw=1.2,zorder=2)
ax[1].axhline(0.704,color='#52514e',lw=0.8,ls='--'); ax[1].text(3.45,0.704,'numerical\nfloor',va='center',ha='left',fontsize=7,color='#52514e')
ax[1].axhline(0.111,color='#8a8984',lw=0.8,ls=':'); ax[1].text(3.45,0.111,'chance',va='center',ha='left',fontsize=7,color='#52514e')
ax[0].set_ylabel('Validation accuracy'); ax[0].set_ylim(0.4,1.02)
ax[1].set_ylabel('Top-20% IoU with parent'); ax[1].set_ylim(0,1.05)
ax[2].set_ylabel('Energy / FP32 parent'); ax[2].set_yscale('log'); ax[2].set_ylim(0.005,2)
ax[2].axhline(1,color='#8a8984',lw=0.6)
for a,t in zip(ax,['(a) Accuracy','(b) Explanation agreement','(c) Estimated energy']):
    a.set_xticks(X); a.set_xticklabels(['FP32','8','4','2']); a.set_xlabel('Bit-width'); a.set_title(t,fontsize=9,loc='left')
    a.grid(axis='y',color='#e4e3df',lw=0.5); a.set_axisbelow(True); a.set_xlim(-0.4,3.4)
ax[2].text(2.15,0.02,'extrapolated\ncosts',fontsize=7,color='#52514e')
ax[2].axvspan(1.5,3.4,color='#f0efeb',zorder=0)
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='outside upper center',ncol=3,frameon=False)
fig.savefig('/home/claude/work/thesis/figures/grid_headline.pdf'); fig.savefig('/tmp/claude-0/fig/grid_headline.png',dpi=160)
for arm in acc: print(arm,{b:round(np.mean(v),4) for b,v in sorted(acc[arm].items())},{b:round(np.mean(v),3) for b,v in sorted(iou[arm].items())},{b:round(np.mean(v),4) for b,v in sorted(en[arm].items())})
