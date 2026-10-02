# New HTTP checks for the post-fix rerun (exec'd inside run_part2.py / part3_http.py; needs U, b64, rec, requests, json).
# Bodies are sent as raw JSON text so "\ud800" reaches the server as a JSON escape (as a browser/client would send it).
def RAW(name, body_text, want, need=None):
    try: r=requests.post(U+"/v1/systemone",data=body_text.encode("ascii"),headers={"Content-Type":"application/json"},timeout=300)
    except Exception as e: rec("[new] HTTP "+name,"FAIL",repr(e)); return None
    ok=r.status_code in want and (need is None or need(r))
    rec("[new] HTTP "+name,"PASS" if ok else "FAIL","status=%s %s"%(r.status_code,r.text[:220])); return r
IMGJ='{"image":"%s"}'%b64
IMGTJ=lambda t:'{"image":"%s","text":%s}'%(b64,t)
QJ=lambda crit,ins='"Which?"':'{"q":{"type":"choice","instructions":%s,"criteria":%s}}'%(ins,crit)
B=lambda st,q:'{"state":%s,"questions":%s}'%(st,q)
SURJ='"a\\ud800b"'
has_sur=lambda r:"surrogate" in r.text
RAW("B1 image: surrogate in option label -> 422",B(IMGJ,QJ('{%s:null,"b":null}'%SURJ)),(422,),has_sur)
RAW("B1 image: surrogate in instructions -> 422",B(IMGJ,QJ('{"a":null,"b":null}',SURJ)),(422,),has_sur)
RAW("B1 image: surrogate in option description -> 422",B(IMGJ,QJ('{"a":%s,"b":null}'%SURJ)),(422,),has_sur)
RAW("B1 image: surrogate in state text -> 422",B(IMGTJ(SURJ),QJ('{"a":null,"b":null}')),(422,),has_sur)
RAW("B1 image: surrogate in state text dict -> 422",B(IMGTJ('{"k":%s}'%SURJ),QJ('{"a":null,"b":null}')),(422,),has_sur)
RAW("B1 text-only: surrogate in str state -> 422",B(SURJ,QJ('{"a":null,"b":null}')),(422,),has_sur)
RAW("B1 text-only: surrogate in dict state value -> 422",B('{"text":%s}'%SURJ,QJ('{"a":null,"b":null}')),(422,),has_sur)
RAW("B1 text-only: surrogate in dict state key -> 422",B('{%s:"x"}'%SURJ,QJ('{"a":null,"b":null}')),(422,),has_sur)
RAW("B1 text-only: surrogate in option label -> 422",B('"hello"',QJ('{%s:null,"b":null}'%SURJ)),(422,),has_sur)
RAW("B1 low surrogate alone \\udc00 in label -> 422",B(IMGJ,QJ('{"x\\udc00":null,"b":null}')),(422,),has_sur)
RAW("B1 control: escaped surrogate PAIR \\ud83d\\udc36 (valid emoji) -> 200",B(IMGJ,QJ('{"\\ud83d\\udc36":null,"b":null}')),(200,),lambda r:"\U0001F436" in r.json()["answers"]["q"]["probabilities"])
RAW("B1 surrogate in question id -> 4xx, not 500",B(IMGJ,'{"q\\ud800":{"type":"choice","instructions":"?","criteria":{"a":null,"b":null}}}'),(400,413,422))
RAW("B1 surrogate in question id, text-only -> 4xx, not 500",B('"hello"','{"q\\ud800":{"type":"choice","instructions":"?","criteria":{"a":null,"b":null}}}'),(400,413,422))
RAW("B1 surrogate in question type -> 4xx",B(IMGJ,'{"q":{"type":"choice\\ud800","instructions":"?","criteria":{"a":null,"b":null}}}'),(400,422))
RAW("B1 surrogate inside image base64 string -> 400",B('{"image":"abc\\ud800"}',QJ('{"a":null,"b":null}')),(400,))
r=requests.get(U+"/health"); rec("[new] HTTP health after surrogate requests","PASS" if r.status_code==200 else "FAIL",r.status_code)
# B2 over HTTP: a GIF whose header says 0x0 (patched from a real 1x1 GIF)
import io as _io
from PIL import Image as _Image
_b=_io.BytesIO(); _Image.new("RGB",(1,1)).save(_b,"GIF"); g=bytearray(_b.getvalue())
g[6:10]=b"\x00\x00\x00\x00"
_i=g.index(b"\x2c"); g[_i+5:_i+9]=b"\x00\x00\x00\x00"
import base64 as _b64m
RAW("B2 0x0 GIF header -> 400 (not 500)",B('{"image":"%s"}'%_b64m.b64encode(bytes(g)).decode(),QJ('{"a":null,"b":null}')),(400,413,422))
# all-caps
CAPSQ=QJ('{"BASSET HOUND":null,"GOLDEN RETRIEVER":null,"PERSIAN CAT":null}','"WHAT BREED IS THIS?"')
r=RAW("CAPS image request returns caller's upper-case labels",B(IMGJ,CAPSQ),(200,),lambda r:list(r.json()["answers"]["q"]["probabilities"])==["BASSET HOUND","GOLDEN RETRIEVER","PERSIAN CAT"] and r.json()["answers"]["q"]["choice"] in ("BASSET HOUND","GOLDEN RETRIEVER","PERSIAN CAT"))
r2=RAW("CAPS lower-case twin request",B(IMGJ,QJ('{"basset hound":null,"golden retriever":null,"persian cat":null}','"what breed is this?"')),(200,))
if r is not None and r2 is not None and r.status_code==r2.status_code==200:
    p1=list(r.json()["answers"]["q"]["probabilities"].values()); p2=list(r2.json()["answers"]["q"]["probabilities"].values())
    rec("[new] HTTP CAPS == lower-case probabilities","PASS" if p1==p2 else "FAIL","caps=%s lower=%s"%(p1,p2))
RAW("CAPS acronym labels DNA/RNA unchanged",B(IMGJ,QJ('{"DNA":null,"RNA":null}','"IS THIS DNA OR RNA?"')),(200,),lambda r:list(r.json()["answers"]["q"]["probabilities"])==["DNA","RNA"])
RAW("CAPS text-only labels unchanged",B('"THE MATCH ENDED 2-1"',QJ('{"SPORTS NEWS":null,"WORLD NEWS":null}','"WHAT TOPIC?"')),(200,),lambda r:list(r.json()["answers"]["q"]["probabilities"])==["SPORTS NEWS","WORLD NEWS"])
