import numpy as np
import sys

ncc_result=np.fromfile('./output/'+sys.argv[1]+'.bin', dtype='float32')
print(ncc_result[0])
print(ncc_result[1])
