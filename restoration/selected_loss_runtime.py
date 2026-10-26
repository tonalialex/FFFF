"""Opt-in full-pipeline loss dispatch; legacy entry remains unchanged by default."""
import os,math
from loss_candidates import CandidateLoss,CONFIGS
_cached={}

def selected_objective(args,teacher_present):
 name=os.environ.get('SAR_RESTORATION_LOSS_CONFIG')
 if not name:return None
 if name not in CONFIGS:raise ValueError('Unknown SAR_RESTORATION_LOSS_CONFIG: '+name)
 g=float(os.environ.get('SAR_RESTORATION_GRADIENT_WEIGHT',CONFIGS[name]['gradient']))
 s=float(os.environ.get('SAR_RESTORATION_STRUCTURE_WEIGHT',CONFIGS[name]['structure']))
 default_teacher={'selected_final':1.0,'l1':.5}.get(name,.1)
 t=float(os.environ.get('SAR_RESTORATION_TEACHER_WEIGHT',default_teacher))
 if not all(math.isfinite(x) and x>=0 for x in [g,s,t]):raise ValueError('Loss weights must be finite and nonnegative.')
 key=(id(args),name,g,s,t)
 if key not in _cached:
  o=CandidateLoss(name,teacher_weight=t);o.config=dict(o.config,gradient=g,structure=s);_cached[key]=o
 args.teacher_weight=t if teacher_present else 0.
 args.restoration_loss_recipe={'name':name,'gradient':g,'structure':s,'teacher_effective':args.teacher_weight,
  'alignment_radius':_cached[key].radius,'supervised_margin':4,'enabled_by':'run.py'}
 return _cached[key]
