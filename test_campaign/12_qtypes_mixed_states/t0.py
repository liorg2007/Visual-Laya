import sys,json;sys.path.insert(0,'/home/zera/laya-vision')
import laya_vision,torch
a=laya_vision.load('/home/zera/laya-vision/runs/stage2_c/wise085_calibrated',device='cuda')
r=a.predict({"image":"/home/zera/laya-vision/data/images/coco/1000.jpg"},{"c":{"type":"choice","instructions":"What is shown?","criteria":{"A":"a dog","B":"a bus"}},
"s":{"type":"score","instructions":"How crowded is the scene?","criteria":["empty","few","many"]},"n":{"type":"noul","instructions":"Is there a person?"}})
print(json.dumps(r,indent=1)[:2500])
