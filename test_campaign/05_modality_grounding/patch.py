s=open('run.py').read()
s=s.replace('CONDS = ["real", "black", "white", "gray", "noise", "patch4x4", "patch8x8", "shuffle_within", "swap_cross_task", "text_only"]',
 'CONDS = ["real", "black", "white", "noise", "patch4x4", "shuffle_within", "swap_cross_task", "text_only"]\ntorch.set_num_threads(4)\nDEADLINE = float(os.environ.get("BUDGET", 2400))')
a=s.index('rows = []\nt0')
head=s[:a]; body=s[a:]
body=body.replace('''for t in TASKS:
    for im, qs in data[t]:
''','''order = []
for r in range(240):
    for t in TASKS:
        k = 4 if t == "vqav2_yesno" else 1
        for j in range(k):
            i = r * k + j
            if i < len(data[t]): order.append((t,) + data[t][i])
for t, im, qs in order:
    if True:
        if time.time() - t0 > DEADLINE: print("deadline hit", flush=True); break
''')
body=body.replace('    print(t, "done", round(time.time() - t0), "s", flush=True)\n','')
body=body.replace('            if nst % 50 == 0: print("states", nst, round(time.time() - t0), "s", flush=True)','            if nst % 40 == 0:\n                print("states", nst, round(time.time() - t0), "s", flush=True); json.dump(rows, open("%s/raw_%s.json" % (OUT, TAG), "w"))')
open('run.py','w').write(head+body)
