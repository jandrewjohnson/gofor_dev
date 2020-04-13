# -*- mode: python -*-

block_cipher = None



import sys
import shutil
import os
from PyInstaller.compat import is_win

# Global Variables
current_dir = os.path.join(os.getcwd(), os.path.dirname(sys.argv[1]))

print('CURRENT DIR:', current_dir)

# # Analyze Scripts for Dependencies
# # Add the release virtual environment to the extended PATH.
# # This helps IMMENSELY with trying to get the binaries to work from within
# # a virtual environment, even if the virtual environment is hardcoded.
# path_extension = []
# if is_win:
#     path_base = os.path.join('mesh_env', 'lib')
# else:
#     path_base = os.path.join('mesh_env', 'lib', 'python2.7')
# path_base = os.path.abspath(path_base)
# path_extension.insert(0, path_base)
# path_extension.insert(0, current_dir)
# path_extension.insert(0, os.path.join(path_base, 'site-packages'))
# path_extension.insert(0, 'code')
# print 'PATH EXT: %s' % path_extension



kwargs = {
    'hookspath': [os.path.join(current_dir, 'build/hooks')],
    'hiddenimports': [
        'hazelbean',
        'geopandas',
        'markdown',
        'pandas',
        'fiona',
        'distutils',
        'distutils.dist',
        'natcap.versioner',
        'natcap.versioner.version',
        'natcap.invest.version',
        'pygeoprocessing.version',
        'fiona',
        'fiona._spec',
        'fiona.scheme',
        'netCDF4',
        'sklearn',
        'statsmodels',
    ],
}





datas = [('C:\Anaconda\Lib\site-packages\osgeo\gdal300.dll','gdal')]
datas.append(('C:\Anaconda\Lib\site-packages\osgeo\geos_c.dll','geos'))
datas.append(('C:\\Anaconda\\pkgs\\qt-5.6.2-vc14h6f8c307_12\\Library\\plugins\\platforms','platforms'))

# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = resourcePath("qt5_plugins")

a = Analysis(['gofor_model.py'],
             pathex=['c:\\OneDrive\\Projects\\gofor\\src', 'c:\\OneDrive\\Projects\\gofor', 'C:\\Anaconda\\pkgs\\qt-5.6.2-vc14h6f8c307_12\\Library\\plugins\\platforms'],
             datas=datas,
             binaries=[],
             runtime_hooks=[],
             win_no_prefer_redirects=False,
             excludes=[],
             win_private_assemblies=False,
             cipher=block_cipher,
             **kwargs)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='gofor',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='gofor')
