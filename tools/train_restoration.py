"""Standalone restoration entry; defaults to the final selected loss."""
from pathlib import Path
import argparse,sys,json,math
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'registration'),str(ROOT/'restoration')]

def main():
 from loss_candidates import CandidateLoss,CONFIGS
 import train_pair_restoration as training
 p=argparse.ArgumentParser(add_help=False)
 p.add_argument('--loss-config',choices=list(CONFIGS),default='selected_final')
 p.add_argument('--gradient-weight',type=float)
 p.add_argument('--structure-weight',type=float)
 p.add_argument('--alignment-radius',type=int,default=3)
 known,remaining=p.parse_known_args()
 if known.alignment_radius not in [0,1,2,3,4]:raise ValueError('alignment-radius must be between 0 and 4.')
 a=training.build_parser().parse_args(remaining)
 selection=json.loads((ROOT/'configs/loss_selected.json').read_text())
 if a.init_checkpoint and '--teacher-weight' not in remaining:
  a.teacher_weight=selection['teacher'] if known.loss_config=='selected_final' else .1
 objective=CandidateLoss(known.loss_config,radius=known.alignment_radius)
 objective.config=dict(objective.config)
 for arg,key in [(known.gradient_weight,'gradient'),(known.structure_weight,'structure')]:
  if arg is not None:
   if not math.isfinite(arg) or arg<0:raise ValueError(key+' must be finite and nonnegative.')
   objective.config[key]=arg
 if not math.isfinite(a.teacher_weight) or a.teacher_weight<0:raise ValueError('teacher-weight')
 training._training_loss=objective
 report=training.train_pair_restoration(a)
 out=Path(a.output_dir)
 recipe={'name':known.loss_config,'configuration':objective.config,'alignment_radius':objective.radius,
  'teacher_weight':a.teacher_weight if a.init_checkpoint else 0,'supervised_margin':4,
  'scope':'Final selected single-pair weights; not a new weight search.','report':report}
 (out/'selected_loss_recipe.json').write_text(json.dumps(recipe,indent=2))

if __name__=='__main__':main()
