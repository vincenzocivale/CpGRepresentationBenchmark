"""Download official EWAS Catalog community deposits, with checksums and archive validation.

Two deposits outside the five requested families (neurological dementia and
untargeted circulating proteome) retain their official metadata but not results.
"""
import json,time,hashlib,gzip,zipfile,requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path('data/raw/ewas_catalog_expansion/zenodo')
root.mkdir(parents=True, exist_ok=True)
response=requests.get('https://zenodo.org/api/communities/ewas-catalog/records', params={'size':25},timeout=60)
response.raise_for_status()
data=response.json()
records=data['hits']['hits']
total=data['hits']['total']
if isinstance(total,dict): total=total['value']
if total != len(records): raise RuntimeError('Community pagination required; refusing incomplete inventory')
(root/'community_records.json').write_text(json.dumps(data,indent=2))
for r in records:
 d=root/str(r['id']);d.mkdir(exist_ok=True);(d/'record.json').write_text(json.dumps(r,indent=2))
jobs=[(r,f) for r in records if r['id'] not in [6801458,10359769] for f in r.get('files',[])]
def get(job):
 r,f=job;d=root/str(r['id']);p=d/f['key'];tmp=p.with_suffix(p.suffix+'.part')
 for i in range(9):
  try:
   h=hashlib.md5()
   if not p.exists():
    offset=tmp.stat().st_size if tmp.exists() else 0
    url=f"https://zenodo.org/records/{r['id']}/files/{f['key']}?download=1"
    if f['size'] >= 100_000_000:
     def part(bounds):
      a,b=bounds
      for retry in range(9):
       try:
        res=requests.get(url,headers={'Range':f'bytes={a}-{b}'},timeout=(20,120))
        res.raise_for_status()
        if res.status_code!=206 or len(res.content)!=b-a+1:raise ValueError('Invalid range response')
        if not res.headers.get('Content-Range','').startswith(f'bytes {a}-{b}/'):raise ValueError('Range start mismatch')
        return res.content
       except Exception:
        if retry==8:raise
        time.sleep(10)
     with open(tmp,'ab') as out, ThreadPoolExecutor(4) as parts:
      while offset < f['size']:
       bounds=[(a,min(a+(8<<20)-1,f['size']-1)) for a in range(offset,min(offset+(32<<20),f['size']),8<<20)]
       for b in parts.map(part,bounds):out.write(b);offset+=len(b)
    else:
     with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(20,120)) as res:
      res.raise_for_status()
      with open(tmp,'ab' if offset and res.status_code==206 else 'wb') as out:
       for b in res.iter_content(4<<20):out.write(b)
    tmp.replace(p)
   with open(p,'rb') as inp:
    while b:=inp.read(4<<20):h.update(b)
   if 'md5:'+h.hexdigest()!=f['checksum']:
    p.unlink()
    raise ValueError('MD5 mismatch; discard invalid file and retry')
   sha=hashlib.sha256()
   with open(p,'rb') as inp:
    while b:=inp.read(4<<20):sha.update(b)
   if p.name.endswith('.gz'):
    with gzip.open(p,'rb') as inp:
     while inp.read(4<<20):pass
   if p.suffix=='.zip':
    with zipfile.ZipFile(p) as z:assert z.testzip() is None
   info={'status':'verified','url':f['links']['self'],'expected_md5':f['checksum'],'sha256':sha.hexdigest(),'bytes':p.stat().st_size}
   p.with_suffix(p.suffix+'.verified.json').write_text(json.dumps(info,indent=2));print('VERIFIED',r['id'],f['key'],p.stat().st_size,flush=True);return info
  except Exception as e:
   print('RETRY',r['id'],f['key'],i+1,str(e),flush=True)
   if i<8:time.sleep(10)
 return {'status':'failed','record':r['id'],'file':f['key']}
with ThreadPoolExecutor(4) as ex:results=list(ex.map(get,jobs))
(root/'download_manifest.json').write_text(json.dumps(results,indent=2))
