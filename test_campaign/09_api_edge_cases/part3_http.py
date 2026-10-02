import io, os, json, base64, time, subprocess, sys, requests, numpy as np
from PIL import Image
R="/home/zera/laya-vision"; CK=R+"/runs/stage2_c/wise085_calibrated"; IMG=R+"/data/images/oxford_pets/basset_hound_129.jpg"
results=[]
def rec(n,s,d): results.append({"test":n,"status":s,"detail":str(d)[:500]}); print(s,n,str(d)[:300],flush=True)
t0=time.time()
p=subprocess.Popen([R+"/.venv/bin/python","-m","laya_vision.ui","--checkpoint",CK,"--port","8793","--device","cuda"],stdout=open("ui_server.log","w"),stderr=subprocess.STDOUT)
U="http://127.0.0.1:8793"
for _ in range(1500):
    try: requests.get(U+"/health",timeout=1); break
    except Exception: time.sleep(1)
rec("CLI python -m laya_vision.ui starts and serves /health","PASS",time.time()-t0)
def enc(im,fmt,**kw):
    b=io.BytesIO(); im.save(b,fmt,**kw); return base64.b64encode(b.getvalue()).decode()
Q={"q":{"type":"choice","instructions":"What animal?","criteria":{"cat":None,"dog":None}}}
bim=Image.open(IMG).convert("RGB")
def H(name,body,want=(200,)):
    try: r=requests.post(U+"/v1/systemone",json=body,timeout=900)
    except Exception as e: rec("HTTP "+name,"FAIL",repr(e)); return
    rec("HTTP "+name,"PASS" if r.status_code in want else "FAIL","status=%s %s"%(r.status_code,r.text[:200]))
t=time.time()
H("first request (cold)",{"state":{"image":enc(bim,"JPEG")},"questions":Q}); print("cold s",time.time()-t)
H("PNG",{"state":{"image":enc(bim,"PNG")},"questions":Q})
H("WEBP",{"state":{"image":enc(bim,"WEBP")},"questions":Q})
H("GIF",{"state":{"image":enc(bim.convert("P"),"GIF")},"questions":Q})
H("BMP (server allows JPEG/PNG/WEBP/GIF only) expect 400",{"state":{"image":enc(bim,"BMP")},"questions":Q},(400,))
H("TIFF expect 400",{"state":{"image":enc(bim,"TIFF")},"questions":Q},(400,))
H("RGBA PNG",{"state":{"image":enc(bim.convert("RGBA"),"PNG")},"questions":Q})
H("grayscale PNG",{"state":{"image":enc(bim.convert("L"),"PNG")},"questions":Q})
H("palette PNG",{"state":{"image":enc(bim.convert("P"),"PNG")},"questions":Q})
H("CMYK JPEG",{"state":{"image":enc(bim.convert("CMYK"),"JPEG")},"questions":Q})
H("1x1 PNG",{"state":{"image":enc(Image.new("RGB",(1,1)),"PNG")},"questions":Q})
H("non-square 4000x100 PNG",{"state":{"image":enc(bim.resize((4000,100)),"PNG")},"questions":Q})
H("7000x7000 PNG expect 413",{"state":{"image":enc(Image.new("RGB",(7000,7000)),"PNG")},"questions":Q},(413,))
H("max_len=64 (exactly fits) 200",{"state":{"image":enc(bim,"JPEG")},"questions":Q,"max_len":64})
H("max_len=40 image does not fit expect 422",{"state":{"image":enc(bim,"JPEG")},"questions":Q,"max_len":40},(422,))
H("head_max_len=8",{"state":{"image":enc(bim,"JPEG")},"questions":Q,"head_max_len":8},(200,400,422))
H("64 questions (server cap)",{"state":{"image":enc(bim,"JPEG")},"questions":{"q%d"%i:Q["q"] for i in range(64)}})
H("65 questions expect 413",{"state":{"image":enc(bim,"JPEG")},"questions":{"q%d"%i:Q["q"] for i in range(65)}},(413,))
H("options 100 (limit?)",{"state":{"image":enc(bim,"JPEG")},"questions":{"q":{"type":"choice","instructions":"?","criteria":{"o%d"%i:None for i in range(100)}}}},(200,400,413,422))
H("options 300",{"state":{"image":enc(bim,"JPEG")},"questions":{"q":{"type":"choice","instructions":"?","criteria":{"o%d"%i:None for i in range(300)}}}},(200,400,413,422))
H("very long option text 60k chars",{"state":{"image":enc(bim,"JPEG")},"questions":{"q":{"type":"choice","instructions":"?","criteria":{"a":"x"*60000,"b":None}}}},(200,400,413,422))
H("noul question",{"state":{"image":enc(bim,"JPEG")},"questions":{"q":{"type":"noul","instructions":"Is this a dog?","criteria":{"false":"no","true":"yes"}}}})
H("score question",{"state":{"image":enc(bim,"JPEG")},"questions":{"q":{"type":"score","instructions":"How cute is it?"}}},(200,400,422))
H("lone surrogate in options (JSON) expect 422",{"state":{"image":enc(bim,"JPEG")},"questions":{"q":{"type":"choice","instructions":"?","criteria":{"a\ud800":None,"b":None}}}},(422,))
r=requests.post(U+"/v1/systemone",data=b'{"state":"hi","questions":{"q":{"type":"choice","instructions":"x","criteria":{"a":null,"b":null}}},"state":"dup"}',headers={"Content-Type":"application/json"}); rec("HTTP duplicate JSON key","PASS" if r.status_code<500 else "FAIL",r.status_code)
r=requests.post(U+"/v1/systemone",data=b'{"state":NaN,"questions":{}}',headers={"Content-Type":"application/json"}); rec("HTTP NaN literal in JSON","PASS" if r.status_code<500 else "FAIL","%s %s"%(r.status_code,r.text[:150]))
r=requests.post(U+"/v1/systemone",data=("["*5000).encode(),headers={"Content-Type":"application/json"}); rec("HTTP deeply nested JSON (no 500)","PASS" if r.status_code<500 else "FAIL","%s %s"%(r.status_code,r.text[:150]))
r=requests.options(U+"/v1/systemone"); rec("HTTP OPTIONS (CORS) info","PASS",r.status_code)
r=requests.get(U+"/"); rec("UI page contains 'systemone' fetch and file input","PASS" if ("input" in r.text and "systemone" in r.text and "file" in r.text) else "FAIL",len(r.text))
b64=enc(bim,"JPEG")
exec(open("post_fix_http.py").read())
r=requests.get(U+"/health"); rec("health ok after abuse","PASS" if r.status_code==200 else "FAIL",r.status_code)
p.terminate(); p.wait(30); rec("server process exited","PASS" if p.poll() is not None else "FAIL","returncode=%s"%p.poll())
json.dump(results,open("results_part3.json","w"),indent=1); print("TOTAL",len(results))
