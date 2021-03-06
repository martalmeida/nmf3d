from . import calcs
from . import vertical_structure
from . import hough_functions
from . import expansion_coeffs
from . import inv_expansion_coeffs
from .version import __version__

def __folders():
  import os
  here=__path__[0]
  return os.path.join(here,'data'),os.path.join(here,'doc')

#datafolder,docfolder=__folders()
docfolder=__folders()[-1]

def doc():
  url='https://github.com/martalmeida/nmf3d'
  urldoc=url+'/tree/master/okean/doc/'

  import glob
  import os
  files=glob.glob(os.path.join(docfolder,'*'))
  files.sort()
  if len(files)==1:
    print('Found doc file %s'%files[0])
    import webbrowser
    furl=os.path.join(urldoc,os.path.basename(files[0]))
    webbrowser.open(furl)
  else:
    print('Available doc files:')
    for f in files: print(' - %s'%f)
    print('Doc files also available in %s'%urldoc)
