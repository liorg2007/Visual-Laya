import numpy as np, json
from laya_vision.images import load_image
out={}
for n,a in [("0x0x3",np.zeros((0,0,3),np.uint8)),("0x5",np.zeros((0,5),np.uint8)),("5x0x3",np.zeros((5,0,3),np.uint8))]:
    try: im=load_image(a); out[n]="no error, size=%s mode=%s"%(im.size,im.mode)
    except Exception as e: out[n]="%s: %s"%(type(e).__name__,e)
print(json.dumps(out,indent=1)); json.dump(out,open("probe_images.json","w"),indent=1)
