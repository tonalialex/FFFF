"""Loss-only alternatives; alignment is estimated from input and detached target."""
import torch
import torch.nn.functional as F

def charbon(a,b):
    return (torch.sqrt((a-b).square()+1e-6)-1e-3).mean()

def smooth(a):
    return F.avg_pool2d(a,3,stride=1,padding=1)

def grad(a):
    a=smooth(a)
    return a[...,1:,:]-a[...,:-1,:],a[...,:,1:]-a[...,:,:-1]

def gradient_loss(a,b):
    ax,ay=grad(a);bx,by=grad(b)
    return .5*(charbon(ax,bx)+charbon(ay,by))

def lncc(a,b):
    # Target-derived support prevents near-black sea dominating the score.
    k=11
    avg=lambda x:F.avg_pool2d(x,k,1,k//2)
    ma,mb=avg(a),avg(b)
    va=(avg(a*a)-ma*ma).clamp_min(0)
    vb=(avg(b*b)-mb*mb).clamp_min(0)
    cov=avg(a*b)-ma*mb
    corr=(cov/torch.sqrt((va+1e-4)*(vb+1e-4))).clamp(-1,1)
    support=(avg(b.detach())>.025).float()
    return ((1-corr)*support).sum()/support.sum().clamp_min(1)

@torch.no_grad()
def aligned_target(source,target,radius=3):
    """One bounded integer translation per patch, no prediction-driven registration.

    Every objective uses the same central support (four-pixel margin).
    No roll/wraparound or padded pixels enter supervision.
    """
    margin=4
    b,_,h,w=source.shape;hh,ww=h-2*margin,w-2*margin
    x=F.avg_pool2d(source,5,1,2)[...,margin:-margin,margin:-margin].flatten(1)
    y=F.avg_pool2d(target,5,1,2)
    candidates=F.unfold(y,kernel_size=(hh,ww))
    x=x-x.mean(1,keepdim=True)
    yc=candidates-candidates.mean(1,keepdim=True)
    scores=(x[:,:,None]*yc).sum(1)/(x.norm(dim=1)[:,None]*yc.norm(dim=1)+1e-8)
    yy,xx=torch.meshgrid(torch.arange(2*margin+1,device=x.device),torch.arange(2*margin+1,device=x.device),indexing='ij')
    dx=(xx-margin).flatten();dy=(yy-margin).flatten()
    allowed=(dx.abs()<=radius)&(dy.abs()<=radius)
    scores=scores-.002*(dx.square()+dy.square())[None,:]
    scores[:,~allowed]=-1e6
    values,best=scores.max(1)
    best=torch.where(values>.25,best,torch.full_like(best,margin*(2*margin+1)+margin))
    raw=F.unfold(target,kernel_size=(hh,ww))
    chosen=raw.gather(2,best[:,None,None].expand(-1,hh*ww,1)).reshape(b,1,hh,ww)
    shifts=torch.stack([dx[best],dy[best]],dim=1)
    return chosen,shifts,values

CONFIGS={
 'l1':dict(align=False,pixel='l1',gradient=0.,structure=0.),
 'charbon':dict(align=False,pixel='charbon',gradient=0.,structure=0.),
 'charbon_grad':dict(align=False,pixel='charbon',gradient=.1,structure=0.),
 'align_charbon':dict(align=True,pixel='charbon',gradient=0.,structure=0.),
 'align_grad005':dict(align=True,pixel='charbon',gradient=.05,structure=0.),
 'align_grad015':dict(align=True,pixel='charbon',gradient=.15,structure=0.),
 'align_grad_ncc':dict(align=True,pixel='charbon',gradient=.1,structure=.02),
 'align_grad015_r2':dict(align=True,pixel='charbon',gradient=.15,structure=0.,radius=2),
 'align_grad015_r4':dict(align=True,pixel='charbon',gradient=.15,structure=0.,radius=4),
 # Current default weights specified by the user on 2026-10-20.
 'selected_final':dict(align=True,pixel='charbon',gradient=0.2,structure=0.3),
}

class CandidateLoss:
    def __init__(self,name,radius=3,teacher_weight=None):
        self.name=name;self.config=CONFIGS[name];self.radius=self.config.get('radius',radius)
        self.shift_counts={};self.teacher_weight=teacher_weight
    def __call__(self,prediction,source,target,args,teacher_prediction=None):
        cfg=self.config
        pred=prediction[...,4:-4,4:-4]
        if cfg['align']:
            target,shifts,quality=aligned_target(source,target,self.radius)
            for shift in shifts.cpu().tolist():
                key=','.join(map(str,shift));self.shift_counts[key]=self.shift_counts.get(key,0)+1
        else:target=target[...,4:-4,4:-4]
        pixel=F.l1_loss(pred,target) if cfg['pixel']=='l1' else charbon(pred,target)
        total=pixel;parts={'pixel':float(pixel.detach())}
        if cfg['gradient']:
            edge=gradient_loss(pred,target);total=total+cfg['gradient']*edge;parts['gradient']=float(edge.detach())
        if cfg['structure']:
            structural=lncc(pred,target);total=total+cfg['structure']*structural;parts['lncc']=float(structural.detach())
        weight=args.teacher_weight if self.teacher_weight is None else self.teacher_weight
        if teacher_prediction is not None and weight>0:
            teacher=F.l1_loss(pred,teacher_prediction.detach()[...,4:-4,4:-4])
            total=total+weight*teacher;parts['teacher']=float(teacher.detach())
        return total,parts

def verify(device='cuda'):
    torch.manual_seed(7)
    x=torch.rand(2,1,32,32,device=device)
    y=torch.roll(x,(2,-1),(-2,-1))
    t,shifts,_=aligned_target(x,y,3)
    assert torch.equal(shifts,torch.tensor([[-1,2],[-1,2]],device=device)),shifts
    assert torch.equal(t,x[...,4:-4,4:-4])
    class Args:teacher_weight=0
    reports={}
    for name in CONFIGS:
        p=(x*.9).detach().requires_grad_()
        loss,parts=CandidateLoss(name)(p,x,y,Args())
        loss.backward();assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
        reports[name]={'loss':float(loss.detach()),'gradient_norm':float(p.grad.norm())}
    return {'passed':True,'known_translation_recovered':[-1,2],'finite_gradients':reports}
