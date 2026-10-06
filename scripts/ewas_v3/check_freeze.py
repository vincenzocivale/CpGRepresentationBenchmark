#!/usr/bin/env python3
"""Mandatory preflight before EWAS-v3 model execution; approval is an external user decision."""
import hashlib,json,subprocess
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
TAG='ewas-v3-main-panel-freeze-v1'
REL='configs/ewas_v3/main_trait_panel.yaml'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1<<22):h.update(b)
 return h.hexdigest()
def main():
 manifest_bytes=(ROOT/REL).read_bytes()
 committed=subprocess.check_output(['git','show',f'{TAG}:{REL}'],cwd=ROOT)
 if committed!=manifest_bytes:raise SystemExit('Manifest differs from committed freeze tag.')
 m=yaml.safe_load(manifest_bytes)
 pins={**m['provenance_sha256'],**m['artifact_sha256']}
 for t in m['main_tasks']:
  for s in t['sets'].values():pins[s['positive_set']]=s['positive_sha256']
 pins['data/derived/ewas_v3/background_exclusion_mask.npy']=m['matching']['mask_sha256']
 for p,expected in pins.items():
  if sha(ROOT/p)!=expected:raise SystemExit(f'Frozen input/artifact changed: {p}')
 approval=ROOT/'configs/ewas_v3/user_approval.json'
 if not approval.exists():raise SystemExit('Panel integrity verified. STOP: explicit user approval has not been recorded; no representation may run.')
 a=json.loads(approval.read_text())
 if a.get('manifest_sha256')!=hashlib.sha256(manifest_bytes).hexdigest() or a.get('approved') is not True or not a.get('user_message'):
  raise SystemExit('Approval must record the actual user instruction and this exact manifest SHA256.')
 print('Frozen panel and explicit approval verified.')
if __name__=='__main__':main()
