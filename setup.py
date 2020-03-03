'''tridimensional normal mode functions'''

classifiers = '''\
License :: European Union Public Licence - EUPL v.1.1
Programming Language :: Python
Topic :: Scientific/Engineering
'''

import glob
#data=glob.glob('data/*') # data is now on a separated repo: nmf3d_data
doc=glob.glob('doc/*.ipynb')
mat=glob.glob('nmf3d_mat/*')

if __name__ == '__main__':
    from numpy.distutils.core import setup

    exec(open('nmf3d/version.py').read()),
    setup(name = "nmf3d",
    version=__version__,
    long_description = __doc__,
    author = "M Marta-Almeida, C A F Marques, J Castanheira",
    author_email = "m.martalmeida@gmail.com",
    url = "https://github.com/martalmeida/nmf3d",
    packages = ['nmf3d'],
    license = 'EUPL',
    platforms = ["any"],
    data_files = [('nmf3d/doc', doc),
#                  ('nmf3d/data',data),
                  ('nmf3d/nmf3d_mat',mat)
                 ]
    )
