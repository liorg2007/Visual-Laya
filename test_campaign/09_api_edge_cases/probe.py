import laya_vision, json
a=laya_vision.load("/home/zera/laya-vision/runs/stage2_c/wise085_calibrated",device="cuda")
print(a.cfg["vision"], a.cfg.get("max_len"), a.cfg.get("head_max_len"), a.n_image_tokens)
q={"q":{"type":"choice","instructions":"What animal?","criteria":{"cat":None,"dog":None}}}
print(json.dumps(a.predict({"image":"/home/zera/laya-vision/data/images/oxford_pets/basset_hound_129.jpg"},q)))
import inspect, laya.agent as la
print(inspect.signature(a.predict), inspect.signature(a.predict_batch))
