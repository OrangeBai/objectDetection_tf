import sys
import os
print('Python %s on %s' % (sys.version, sys.platform))
cur_pwd = os.getcwd()
sys.path.extend([cur_pwd])