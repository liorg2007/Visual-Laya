import json, numpy as np, sys
def agg(rounds, key, f="p50"):
    v=[r[key][f] for r in rounds if key in r]; return np.array(v)
def summarize(path):
    d=json.load(open(path)); rs=d["rounds"]; keys=[k for k in rs[0] if isinstance(rs[0][k],dict)]
    out={}
    for k in keys:
        p50=agg(rs,k,"p50"); p95=agg(rs,k,"p95")
        out[k]={"p50_med_of_rounds":float(np.median(p50)),"p50_min":float(p50.min()),"p50_max":float(p50.max()),
                "p95_med_of_rounds":float(np.median(p95)),"p95_max":float(p95.max()),"n_rounds":len(p50),"n_per_round":rs[0][k]["n"],
                "peak_MB":rs[-1][k].get("peak_MB")}
    return d,out
if __name__=="__main__":
    d,o=summarize(sys.argv[1])
    for k,v in o.items(): print("%-22s p50 %7.1f [%7.1f-%7.1f]  p95 %7.1f (max %7.1f)  n=%dx%d peak=%s"%(k,v["p50_med_of_rounds"],v["p50_min"],v["p50_max"],v["p95_med_of_rounds"],v["p95_max"],v["n_rounds"],v["n_per_round"],v["peak_MB"]))
