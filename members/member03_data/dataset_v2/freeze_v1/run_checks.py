"""Safe acceptance command runner; no model execution, never creates a second dataset."""
import json,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parents[1];R=D.parents[2];O=D/'freeze_v1';rel=str(D.relative_to(R));py='/usr/bin/python3'
commands=[('regression',[py,'-B','-m','unittest','discover','-s',rel,'-p','test_acceptance.py','-v'],0),('plan',[py,'-B',rel+'/build_dataset_v2.py','--plan'],0),('existing_build_refusal',[py,'-B',rel+'/build_dataset_v2.py','--build'],1),('final_verifier',[py,'-B',rel+'/verify_dataset_v2.py'],0),('independent_plan',[py,'-B',rel+'/plan_recheck.py'],0),('sha256sum',['sha256sum','-c','SHA256SUMS.txt'],0)]
results=[]
for name,args,expected in commands:
 start=time.time();cwd=D if name=='sha256sum' else R
 with (O/(name+'.log')).open('w') as f:p=subprocess.run(args,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
 results.append({'name':name,'command':args,'cwd':str(cwd),'exit_code':p.returncode,'expected_exit_code':expected,'seconds':round(time.time()-start,3),'passed':p.returncode==expected,'log':name+'.log'})
 (O/'commands.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print(name,p.returncode,flush=True)
 if p.returncode!=expected:raise SystemExit(1)
print('ALL_SAFE_COMMANDS_EXPECTED_RESULTS')
