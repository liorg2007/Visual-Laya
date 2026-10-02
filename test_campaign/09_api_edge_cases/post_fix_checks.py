# New checks added for the post-fix rerun (exec'd at the end of run_api.py; uses its a, S, Q, run, rec, ...).
# All test names start with "[new]" so the original check set can be compared before/after.
import laya_vision.agent as LA, laya_vision.textnorm as TN
from laya_vision.images import load_image, ImageError
from laya_vision.sequence import to_internal
SUR="a\ud800b"
def strict_sur(name, fn):
    """B1: must raise ValueError whose message names 'surrogate'."""
    try:
        v=fn(); rec("[new] B1 "+name,"FAIL","no error: %s"%str(v)[:200])
    except Exception as e:
        ok=isinstance(e,ValueError) and "surrogate" in str(e)
        rec("[new] B1 "+name,"PASS" if ok else "FAIL","%s: %s"%(type(e).__name__,str(e)[:250]))
def QQ(crit,ins="Which?"): return {"q":{"type":"choice","instructions":ins,"criteria":crit}}
strict_sur("image: surrogate in option label",lambda:a.predict(S,QQ({SUR:None,"b":None})))
strict_sur("image: surrogate in instructions",lambda:a.predict(S,QQ({"a":None,"b":None},ins=SUR)))
strict_sur("image: surrogate in option description",lambda:a.predict(S,QQ({"a":SUR,"b":None})))
strict_sur("image: surrogate in state text (str)",lambda:a.predict({"image":IMG,"text":SUR},Q()))
strict_sur("image: surrogate in state text (dict value)",lambda:a.predict({"image":IMG,"text":{"k":SUR}},Q()))
strict_sur("image: surrogate in state text (list)",lambda:a.predict({"image":IMG,"text":["ok",SUR]},Q()))
strict_sur("text-only: surrogate in str state",lambda:a.predict(SUR,Q()))
strict_sur("text-only: surrogate in dict state value",lambda:a.predict({"text":SUR},Q()))
strict_sur("text-only: surrogate in dict state key",lambda:a.predict({SUR:"x"},Q()))
strict_sur("text-only: surrogate in option label",lambda:a.predict("hello",QQ({SUR:None,"b":None})))
strict_sur("text-only: surrogate in instructions",lambda:a.predict("hello",QQ({"a":None,"b":None},ins=SUR)))
# [test fix] the editor had turned this literal into a real emoji; build the two lone surrogates explicitly
strict_sur("Python str with surrogate pair code points U+D83D U+DC36 (two lone surrogates)",lambda:a.predict(S,QQ({chr(0xD83D)+chr(0xDC36):None,"b":None})))
strict_sur("predict_batch: one bad state among good ones",lambda:a.predict_batch([S,"x\ud800",S],Q()))
strict_sur("second of two questions bad",lambda:a.predict(S,{"ok":Q()["q"],"bad":QQ({SUR:None,"b":None})["q"]}))
def plong():
    try: a.predict_long("x\ud800 "*10,Q()); return (False,"no error")
    except Exception as e: return (isinstance(e,ValueError),"%s: %s"%(type(e).__name__,str(e)[:200]))
run("[new] B1 text-only predict_long with surrogate (expect ValueError)",plong)
run("[new] B1 control: real emoji / non-BMP / NUL / U+FFFD still accepted",lambda:(lambda r:(len(pvec(r))==3,probs(r)))(a.predict({"image":IMG,"text":"\U0001F436 \x00 �"},QQ({"\U0001F436":None,"\U00010348":None,"�":None}))))
run("[new] B1 agent unaffected after surrogate errors",lambda:(lambda d:(d==0,"max|dp|=%g"%d))(max(abs(x-y) for x,y in zip(pvec(a.predict(S,Q())),bp))))
# ---- B2 zero-size images
def b2(name,obj):
    try: im=load_image(obj); rec("[new] B2 "+name,"FAIL","accepted size=%s"%(im.size,))
    except Exception as e: rec("[new] B2 "+name,"PASS" if isinstance(e,ImageError) and "at least one pixel" in str(e) else "FAIL","%s: %s"%(type(e).__name__,e))
b2("load_image ndarray (0,0,3)",np.zeros((0,0,3),np.uint8))
b2("load_image ndarray (0,5)",np.zeros((0,5),np.uint8))
b2("load_image ndarray (5,0,3)",np.zeros((5,0,3),np.uint8))
b2("load_image ndarray (0,0)",np.zeros((0,0),np.uint8))
b2("load_image ndarray CHW (3,0,0)",np.zeros((3,0,0),np.uint8))
b2("load_image ndarray float (0,4,4)",np.zeros((0,4,4)))
b2("load_image PIL 0x4",Image.new("RGB",(0,4)))
b2("load_image PIL 4x0",Image.new("RGB",(4,0)))
b2("load_image PIL 0x0 mode L",Image.new("L",(0,0)))
run("[new] B2 predict with (5,0,3) array -> ValueError (ImageError)",lambda:a.predict({"image":np.zeros((5,0,3),np.uint8)},Q()),"err")
run("[new] B2 control: 1x1 array still accepted",lambda:(lambda r:(math.isclose(sum(pvec(r)),1,abs_tol=2e-3),probs(r)))(a.predict({"image":np.zeros((1,1,3),np.uint8)},Q())))
run("[new] B2 control: 1x1 PIL still accepted",lambda:(lambda r:(math.isclose(sum(pvec(r)),1,abs_tol=2e-3),probs(r)))(a.predict({"image":Image.new("RGB",(1,1))},Q())))
# ---- all-caps handling
calls={"n":0}
_orig=LA.unshout_question
def _spy(q): calls["n"]+=1; return _orig(q)
LA.unshout_question=_spy
CAPS=("BASSET HOUND","GOLDEN RETRIEVER","PERSIAN CAT")
def caps_keys():
    r=a.predict(S,QQ({k:None for k in CAPS},ins="WHAT BREED IS THIS?"))
    ok=list(probs(r))==list(CAPS) and ans(r)["choice"] in CAPS
    return ok,"answer=%s"%json.dumps(ans(r))[:250]
run("[new] CAPS image request returns caller's upper-case labels (keys + choice)",caps_keys)
def caps_eq():
    r1=a.predict(S,QQ({k:None for k in CAPS},ins="WHAT BREED IS THIS?"))
    r2=a.predict(S,QQ({k.lower():None for k in CAPS},ins="what breed is this?"))
    d=max(abs(x-y) for x,y in zip(pvec(r1),pvec(r2)))
    return d==0,"caps=%s lower=%s max|dp|=%g"%(probs(r1),probs(r2),d)
run("[new] CAPS image request == lower-case request (same model input)",caps_eq)
def caps_text():
    r1=a.predict({"image":IMG,"text":"A PHOTO OF MY DOG"},Q()); r2=a.predict({"image":IMG,"text":"a photo of my dog"},Q())
    d=max(abs(x-y) for x,y in zip(pvec(r1),pvec(r2))); return d==0,"max|dp|=%g"%d
run("[new] CAPS image-state text == lower-case text",caps_text)
def acr():
    r=a.predict(S,QQ({"DNA":None,"RNA":None},ins="IS THIS DNA OR RNA?"))
    iq=TN.unshout_question(to_internal(QQ({"DNA":None,"RNA":None},ins="IS THIS DNA OR RNA?")["q"]))
    ok=list(probs(r))==["DNA","RNA"] and list(iq["crit"])==["DNA","RNA"] and iq["ins"]=="is this dna or rna?"
    return ok,"keys=%s model-side crit=%s ins=%r"%(list(probs(r)),list(iq["crit"]),iq["ins"])
run("[new] CAPS acronym-only labels DNA/RNA unchanged (output and model input)",acr)
def acr2():
    r1=a.predict(S,QQ({"DNA":None,"RNA":None},ins="Which molecule?")); r2=a.predict(S,QQ({"dna":None,"rna":None},ins="Which molecule?"))
    d=max(abs(x-y) for x,y in zip(pvec(r1),pvec(r2))); return d>0,"DNA/RNA %s vs dna/rna %s (differ => not rewritten)"%(probs(r1),probs(r2))
run("[new] CAPS DNA/RNA probabilities differ from dna/rna (not rewritten)",acr2)
def acr3():
    out={s:TN.unshout(s) for s in ("USA","CO2","A","UK","DNA","NASA","HTTP","JSON","UNESCO","COVID-19","I AM OK","TOP 10")}
    return all(out[s]==s for s in ("USA","CO2","A","UK","DNA","I AM OK","TOP 10")),json.dumps(out)
run("[new] CAPS short acronyms unchanged; info: 4+ letter acronyms (NASA/HTTP/JSON) are lower-cased for the model",acr3)
def coll():
    r=a.predict(S,QQ({"APPLE":None,"apple":None,"PEAR":None}))
    iq=TN.unshout_question(to_internal(QQ({"APPLE":None,"apple":None,"PEAR":None})["q"]))
    return list(probs(r))==["APPLE","apple","PEAR"] and list(iq["crit"])==["APPLE","apple","PEAR"],"keys=%s model-side=%s"%(list(probs(r)),list(iq["crit"]))
run("[new] CAPS label collision (APPLE vs apple) keeps all labels distinct, unrewritten",coll)
def mixed():
    r=a.predict(S,QQ({"BASSET HOUND":None,"Golden Retriever":None}))
    return list(probs(r))==["BASSET HOUND","Golden Retriever"],"keys=%s"%list(probs(r))
run("[new] CAPS mixed upper/title labels returned unchanged",mixed)
def textonly():
    n0=calls["n"]; r=a.predict("THE MATCH ENDED 2-1",QQ({"SPORTS NEWS":None,"WORLD NEWS":None},ins="WHAT TOPIC IS THIS?")); n1=calls["n"]
    a.predict(S,Q()); n2=calls["n"]
    return n1==n0 and n2>n1 and list(probs(r))==["SPORTS NEWS","WORLD NEWS"],"unshout calls: text-only=%d image=%d; keys=%s"%(n1-n0,n2-n1,list(probs(r)))
run("[new] CAPS text-only request not rewritten (unshout not called), labels kept",textonly)
LA.unshout_question=_orig
