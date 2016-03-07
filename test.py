import init_path
import bing
import numpy as np
import json

w_1st = np.genfromtxt("/home/bbu/Workspace/gazeAssistRecognize/lib/BING-Objectness/doc/weights.txt", delimiter=",").astype(np.float32)
sizes = np.genfromtxt("/home/bbu/Workspace/gazeAssistRecognize/lib/BING-Objectness/doc/sizes.txt", delimiter=",").astype(np.int32)

f = open("/home/bbu/Workspace/gazeAssistRecognize/lib/BING-Objectness/doc/2nd_stage_weights.json")
w_str = f.read()
f.close()
w_2nd = json.loads(w_str)

b = bing.Bing(w_1st,sizes,w_2nd, num_bbs_per_size_1st_stage= 180, num_bbs_final = 5)