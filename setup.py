'''tridimensional normal mode functions'''

import glob
data=glob.glob('data/*')
doc=glob.glob('doc/*')

if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(name = "nmf3d",
    version=0,
    long_description = __doc__,
    author = "M Marta-Almeida, C A F Marques, J Castanheira",
    author_email = "m.martalmeida@gmail.com",
    url = "https://github.com/martalmeida/nmf3d",
    packages = ['nmf3d'],
    platforms = ["any"],
    data_files = [('nmf3d/doc', doc),('nmf3d/data',data)])



