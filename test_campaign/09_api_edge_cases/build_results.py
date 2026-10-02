# Builds results.json (post-fix) from the part logs/JSONs, with before/after counts and a regression diff
# against the pre-fix logs kept in prefix_logs/ (part1_prefix.log = the old run_api.log).
import json, re
def load(p): return json.load(open(p))
def from_log(p):
    out=[]
    for line in open(p, encoding="utf-8", errors="replace"):
        m=re.match(r"^(PASS|FAIL|INFO) (.*)$", line.rstrip("\n"))
        if m: out.append((m.group(1), m.group(2)))
    return out
def counts(rows):
    st=[r["status"] for r in rows]
    return {"n":len(st),"pass":st.count("PASS"),"fail":st.count("FAIL"),"info":len(st)-st.count("PASS")-st.count("FAIL")}
p1=load("results_api.json"); p2=load("results_part2.json"); p3=load("results_part3.json")
p4=load("results_part4.json"); p4b=load("results_part4b.json"); p4c=load("results_part4c.json")
old1=from_log("prefix_logs/part1_prefix.log")
old={"part1":[{"test":t,"status":s} for s,t in old1],  # names matched by prefix below
     "part2":load("prefix_logs/results_part2.json"),"part3":load("prefix_logs/results_part3.json"),
     "part4b":load("prefix_logs/results_part4b.json")}
new={"part1":p1,"part2":p2,"part3":p3,"part4b":p4b}
def status_map(rows): return {r["test"]:r["status"] for r in rows}
regress=[]; fixed=[]
for part in ("part1","part2","part3","part4b"):
    nm=status_map(new[part]); names=sorted(nm,key=len,reverse=True)
    for r in old[part]:
        if part=="part1":  # old log lines are "name detail"; take the longest new name that prefixes it
            hit=next((n for n in names if r["test"]==n or r["test"].startswith(n+" ")),None)
        else: hit=r["test"] if r["test"] in nm else None
        if hit is None: continue
        if r["status"]=="PASS" and nm[hit]!="PASS": regress.append({"part":part,"test":hit,"before":"PASS","after":nm[hit]})
        if r["status"]=="FAIL" and nm[hit]=="PASS": fixed.append({"part":part,"test":hit})
orig1=[r for r in p1 if not r["test"].startswith("[new]")]; new1=[r for r in p1 if r["test"].startswith("[new]")]
orig2=[r for r in p2 if not r["test"].startswith("[new]")]; new2=[r for r in p2 if r["test"].startswith("[new]")]
orig3=[r for r in p3 if not r["test"].startswith("[new]")]; new3=[r for r in p3 if r["test"].startswith("[new]")]
pre=load("results_prefix.json")["parts"]
res={
 "question":"Robustness of the Python API, HTTP /v1/systemone and web UI to edge cases; post-fix rerun verifying B1 (lone surrogate -> 500) and B2 (zero-size image accepted)",
 "checkpoint":"runs/stage2_c/wise085_calibrated",
 "code_under_test":"working tree with uncommitted fixes: laya_vision/textnorm.py (new), agent.py, images.py",
 "parts_before":{k:{kk:v[kk] for kk in ("n","pass","fail") if kk in v} for k,v in pre.items()},
 "parts_after":{
  "part1_python_api":{"all":counts(p1),"original_checks":counts(orig1),"new_checks":counts(new1),"source":"part1.txt / results_api.json"},
  "part2_http_inproc":{"all":counts(p2),"original_checks":counts(orig2),"new_checks":counts(new2),"source":"part2.log / results_part2.json (uvicorn in-process, port 8795)"},
  "part3_http_cli_ui":{"all":counts(p3),"original_checks":counts(orig3),"new_checks":counts(new3),"source":"part3.log / results_part3.json (python -m laya_vision.ui, port 8793, SIGTERM, exited)"},
  "part4_compile":{"all":counts(p4),"source":"part4.log / results_part4.json"},
  "part4b_nocompile":{"all":counts(p4b),"source":"part4b.log / results_part4b.json"},
  "part4c_compile_diag":{"results":p4c,"source":"part4c.log / results_part4c.json"},
 },
 "regressions_prev_pass_now_not":regress,
 "prev_fail_now_pass":fixed,
 "fix_verification":{
  "B1":{"before":"TypeError (Python) / HTTP 500 'inference failed' for surrogate in label, instructions, state text",
        "after":"ValueError naming 'surrogate' (Python, 14/14 checks incl. predict_batch, text-only, dict keys); HTTP 422 with 'surrogate' in detail for label, instructions, option description, image-state text (str/dict), text-only state (str/dict value/dict key), text-only label, lone low surrogate (11/11)",
        "not_covered":["question id with a lone surrogate: HTTP 500 (JSONResponse cannot encode the answer key)","predict_long (text-only) with a surrogate: still TypeError"]},
  "B2":{"before":"(0,0,3)/(0,5)/(5,0,3) arrays accepted as blank images",
        "after":"ImageError 'at least one pixel' for (0,0,3),(0,5),(5,0,3),(0,0),(3,0,0) CHW,(0,4,4) float arrays and PIL 0x4/4x0/0x0; predict raises ImageError (ValueError); 1x1 still accepted; 0x0 GIF over HTTP -> 400"}},
 "caps_checks":"image all-caps request returns caller's upper-case labels and choice, probabilities identical to the lower-case twin (Python and HTTP); DNA/RNA labels unchanged in output and model input; APPLE/apple collision left unrewritten; text-only requests never rewritten. Info: 4+-letter acronyms (NASA, HTTP, JSON, UNESCO) are lower-cased for the model.",
 "remaining_bugs":[
  {"id":"B3","severity":"low","where":"laya_vision/serve.py _inner -> JSONResponse",
   "title":"Lone surrogate in a question id -> HTTP 500 'Internal Server Error' (UnicodeEncodeError while rendering the response, after inference ran)",
   "repro":"curl -X POST /v1/systemone -H 'content-type: application/json' -d '{\"state\":\"hello\",\"questions\":{\"q\\ud800\":{\"type\":\"choice\",\"instructions\":\"?\",\"criteria\":{\"a\":null,\"b\":null}}}}'  (also with an image state). Fix: check_text on question ids too (-> 422)."},
  {"id":"B4","severity":"low","where":"VisionAgent.predict_long -> laya.Agent.predict_long (text-only)",
   "title":"predict_long with a lone surrogate still raises the tokenizer TypeError (B1 fix covers _encode_state only)",
   "repro":"agent.predict_long('x\\ud800', {'q':{'type':'choice','instructions':'?','criteria':{'a':None,'b':None}}}) -> TypeError: TextEncodeInput must be Union[...]. Python-only (not reachable over HTTP)."},
  {"id":"U1","severity":"upstream (not laya_vision)","where":"laya.Agent(compile=True) text predict; torch 2.6.0, transformers 5.17.0, laya 0.3.21",
   "title":"compile=True text-only predict fails in ~7 s with TorchRuntimeError ('NoneType' + FakeTensor at laya/common.py:317, h = h + type_emb). Not a hang. Identical with stock laya.Agent on convaiinnovations/laya, so it is upstream.",
   "repro":"part4c_compile_diag.py"}],
 "test_script_changes":[
  "run_part2.py: defined enc() and base_im (7 checks previously NameError); max_len=64 now expects 200 and a max_len=40 -> 422 check was added; port 8791 -> 8795; exec post_fix_http.py",
  "run_api.py: max_len=64 now expects success (63 tokens) + new max_len=40 -> error check; noul check reads answers['q']['noul'] (was KeyError 'probabilities'); HTTP section removed (duplicate of run_part2.py); 3 load-mode checks at the end are now reached (pre-fix run was interrupted before them); exec post_fix_checks.py",
  "part3_http.py: surrogate-in-option check now strictly expects 422 (was 200/400/422); exec post_fix_http.py; asserts the server process exited",
  "part4_modes.py: compile=True text-only predict re-timed with a 1800 s budget and faulthandler stack dumps; part4c_compile_diag.py added",
  "part4b_nocompile.py: surrogate checks strict (ValueError naming 'surrogate'); image-state text and predict_long cases added",
  "post_fix_checks.py: a surrogate-pair literal had been turned into a real emoji by the editor; now built with chr()"],
 "notes":["No repo source edited, no training.","Pre-fix outputs: results_prefix.json, report_prefix.md, prefix_logs/.",
          "No server or test process left running after the runs (checked ports 8790-8799 and ps)."]
}
json.dump(res,open("results.json","w"),indent=1,ensure_ascii=False)
print(json.dumps({k:v.get("all",v) if isinstance(v,dict) else v for k,v in res["parts_after"].items() if k!="part4c_compile_diag"},indent=0))
print("orig1",counts(orig1),"new1",counts(new1),"orig2",counts(orig2),"new2",counts(new2),"orig3",counts(orig3),"new3",counts(new3))
print("REGRESSIONS",regress); print("FIXED",fixed)
