import logging
from collections import OrderedDict

import numpy as np
import textwrap
import hazelbean as hb
from hazelbean.ui import model, inputs
from hazelbean.ui import validation
import os, sys
import qtpy
from qtpy import QtWidgets
from qtpy import QtCore
from qtpy import QtGui
import time
import six
import qtawesome

logging.basicConfig(level=logging.WARNING)
hb.ui.model.LOGGER.setLevel(logging.WARNING)
hb.ui.inputs.LOGGER.setLevel(logging.WARNING)

L = hb.get_logger('seals')
L.setLevel(logging.INFO)

logging.getLogger('Fiona').setLevel(logging.WARNING)
logging.getLogger('fiona.collection').setLevel(logging.WARNING)

np.seterr(divide='ignore', invalid='ignore')

dev_mode = True
# TODOO NOTE This funtion must be here as it is automatically called by the invest model. Consider making this more flexible in the next release.
@validation.invest_validator
def validate(args, limit_to=None):
    validation_error_list = []
    if not os.path.exists(args['input_lulc_path']):
        validation_error_list.append((['area_of_interest_path'], 'area_of_interest_path did not exist.'))

    if not os.path.exists(args['input_lulc_path']):
        validation_error_list.append((['input_lulc_path'], 'input_lulc_path did not exist.'))

    if not os.path.exists(args['lulc_categorization_table_path']):
        validation_error_list.append((['lulc_categorization_table_path'], 'lulc_categorization_table_path did not exist.'))

    try:
        if not 0.0 < float(args['minimum_patch_size']) <= 1000000:
            validation_error_list.append((['minimum_patch_size'], 'minimum_patch_size must be between 0 and 1000000.'))
    except:
        validation_error_list.append((['minimum_patch_size'], 'minimum_patch_size must be between 0 and 1000000.'))

    try:
        if not 0.0 < float(args['distance_threshold']) <= 1000000:
            validation_error_list.append((['distance_threshold'], 'distance_threshold must be between 0 and 1000000.'))
    except:
        validation_error_list.append((['distance_threshold'], 'distance_threshold must be between 0 and 1000000.'))

    if not os.path.exists(args['lulc_categorization_table_path']):
        validation_error_list.append((['lulc_categorization_table_path'], 'lulc_categorization_table_path did not exist.'))

    return validation_error_list


class GoforSplash(QtWidgets.QDialog):
    """Show a dialog describing InVEST.

    In reasonable accordance with licensing and distribution requirements,
    this dialog not only has information about InVEST and the Natural
    Capital Project, but it also has details about the software used to
    develop and run InVEST and contains links to the licenses for each of
    these other projects.

    Returns:
        None.
    """

    def __init__(self, parent=None):
        """Initialize the AboutDialog.

        Parameters:
            parent=None (QWidget or None): The parent of the dialog.  None if
                no parent.
        """
        QtWidgets.QDialog.__init__(self, parent=parent)
        self.setWindowTitle('GoFor')
        self.setLayout(QtWidgets.QVBoxLayout())
        label_text = """ASDF"""

        # splash_pix = qtpy.QtGui.QPixmap('assets/splash_gofor_small.png')
        # splash = qtpy.QtWidgets.QSplashScreen(splash_pix)
        # splash.show()

        # centered_hbox = QtWidgets.QHBoxLayout()
        # self.setLayout(centered_hbox)
        image_l = QtWidgets.QLabel()
        image_pixmap = qtpy.QtGui.QPixmap('assets/splash_gofor_small.png')
        # scaled_pixmap = image_pixmap
        # if width:
        #     scaled_pixmap = image_pixmap.scaledToWidth(width)
        image_l.setPixmap(image_pixmap)
        # centered_hbox.addWidget(image_l)

        # self.label = QtWidgets.QLabel(label_text)
        # self.label.setTextFormat(QtCore.Qt.RichText)
        # self.label.setOpenExternalLinks(True)
        self.layout().addWidget(image_l)

        self.button_box = QtWidgets.QDialogButtonBox()
        self.accept_button = QtWidgets.QPushButton('OK')
        self.button_box.addButton(
            self.accept_button,
            QtWidgets.QDialogButtonBox.AcceptRole)
        self.layout().addWidget(self.button_box)
        self.accept_button.clicked.connect(self.close)


class GoforUI(model.HazelbeanModel):
    def __init__(self, p):
        self.p = p
        # r"C:\OneDrive\Projects\gofor\branding\splash_gofor.png"
        # splash_pix = qtpy.QtGui.QPixmap('assets/splash_gofor_small.png')
        # splash = qtpy.QtWidgets.QSplashScreen(splash_pix)
        # splash.show()
        # time.sleep(2.5) # LOL, so lazy

        model.HazelbeanModel.__init__(self,
                                   # label=u'seals',
                                   label=u'GoFor: Ecological Restoration Uncertainty Assessment',
                                   target=p.execute,
                                   validator=validate,
                                   localdoc='../documentation/readme.txt')

        self.area_of_interest_path = inputs.File(args_key='area_of_interest_path',
            # default_path=p.default_paths['area_of_interest_path'],
            helptext="A shapefile with a single polygon that will be used to clip any other datasets.",
            label='Area of interest',
            validator=None)
        self.add_input(self.area_of_interest_path)

        self.input_lulc_path = inputs.FileWithDefault(args_key='input_lulc_path',
            helptext=("File path for the land-use, land-cover map to be used as input."),
            label='Land-use, land-cover (raster)',
            validator=validate,
            default_path=os.path.join(p.model_base_data_dir, 'lulc_2015_esa_10s_moll.tif'))
        self.add_input(self.input_lulc_path)

        self.lulc_categorization_table_path = inputs.FileWithDefault(args_key='lulc_categorization_table_path',
            helptext=("File path for the CSV or XLSX file that will defines to which restoration class each LULC class belongs. "
                      "The table is required to have at least 2 columns labeled:"
                      ""
                      " 'lulc_class_id' with integer values representing each value in the LULC raster that should be recategorized. "
                      "Note that if a value exists in the LULC class but not this table, it will default to being considered 'unavailable for restoration'."
                      " and 'restoration_class_id' which will have values of 1 for Forest, 2 Non-forest area potential for restoration, "
                      "or 3 Non-forest areas not suitable for restoration."),
            label='Land-use, land-cover categorization (csv)',
            validator=validate,
            default_path=os.path.join(p.model_base_data_dir, 'lulc_categorization.csv')
        )
        self.add_input(self.lulc_categorization_table_path)


        # NOTE, containers dont need a seperate interactivity slot. has it  by default it seems
        self.advanced_options_container = inputs.Container(
            args_key='advanced_options_container',
            expandable=True,
            expanded=True,
            interactive=True,
            label='Show advanced options')
        self.add_input(self.advanced_options_container)

        current_default = '25.0'
        self.minimum_patch_size = inputs.TextWithDefault(
            args_key='minimum_patch_size',
            helptext=("(hectares)	Minimum patch size: changing the patch size to be different than the default (from the manuscript) value of 100 ha will allow to get more information from finer scale LULC data usually available at the Water Fund scale of work, capturing forests remnants data that were not available with the global datasets used by the authors on their metanalysis"),
            label='Minimum patch size (ha)',
            validator=validate,
            default_text=current_default)



        self.advanced_options_container.add_input(self.minimum_patch_size)
        # self.resampling_threshold = inputs.TextWithDefault(
        #     args_key='resampling_threshold',
        #     helptext=("(meters)	Resampling threshold: having the option to use values less than the original 1km resampling threshold will allow to get more information from finer scale LULC data usually available at the Water Fund scale of work, capturing forests remnants data that were not available with the global datasets used by the authors on their metanalysis"),
        #     label='Resampling threshold',
        #     validator=validate,
        #     default_text='1000.0')
        # self.advanced_options_container.add_input(self.resampling_threshold)

        current_default = '5000.0'
        self.distance_threshold = inputs.TextWithDefault(
            args_key='distance_threshold',
            helptext=("(meters)	Distance threshold: having the option to use values less than the original 5km distance buffer will allow to fine tune the output for specific taxa information (plants, insects, amphibia, birds, mammals) that can be of interest for different purposes and that had to be grouped or neglected in the original authors metanalysis"),
            label='Distance threshold',
            validator=validate,
            default_text=current_default)

        self.advanced_options_container.add_input(self.distance_threshold)

        self.run_clean_intermediate_files = inputs.Checkbox('Remove intermediate files', helptext='help', args_key='run_clean_intermediate_files')
        self.run_clean_intermediate_files.checkbox.setChecked(False)

        self.help_menu.clear()
        self.help_menu.addAction(
            qtawesome.icon('fa.info'),
            'About GoFor', self.about_dialog.exec_)


        self.advanced_options_container.add_input(self.run_clean_intermediate_files)

        for i in reversed(range(self.about_dialog.layout().count())):
            self.about_dialog.layout().itemAt(i).widget().deleteLater()
        self.about_dialog.setWindowTitle('About GoFor')

        self.about_dialog.setLayout(QtWidgets.QVBoxLayout())
        label_text = textwrap.dedent(
            """
            
            
            <h1>GoFor</h1>
            GoFor • Tropical Forests Ecological Restoration Uncertainty Assessment Tool. V 1.0.<br/><br/>
            Developed by:<br/>
Justin Johnson*, Jorge E. Leon Sarmiento**, Julio R. C. Tymus** and Rubens de M. Benini**. 2019.<br/>
*University of Minnesota, **The Nature Conservancy<br/>
 <br/>
Based on:<br/>
Renato Crouzeilles1,2,3, Felipe S. Barros1,4, Paulo G. Molin5, Mariana S. Ferreira6, André B. Junqueira1,2, <br/>
Robin L. Chazdon1,7,8, David B. Lindenmayer9, Julio R. C. Tymus10, Bernardo B. N. Strassburg1,2,3 & Pedro H. S.<br/>
Brancalion11. 2019. A new approach to map landscape variation in forest restoration success at the global scale. 
<br/>Journal of Applied Ecology.<br/>
<br/>
1 International Institute for Sustainability, 22460-320, Rio de Janeiro, Brazil.<br/>
2 Rio Conservation and Sustainability Science Centre, Department of Geography and the Environment, Pontifícia Universidade Católica, 22453-900, Rio de Janeiro, Brazil.<br/>
3 Programa de Pós Graduação em Ecologia, Universidade Federal do Rio de Janeiro, 68020, Rio de Janeiro, Brazil.<br/>
4 Reference Center on Technological Information and Management system with Free Software (CeRTIG + SoL), National University of Misiones, 3300, Posadas, Argentina.<br/>
5 Center for Nature Sciences, Federal University of São Carlos, 18245-970, São Carlos, Brazil.<br/>
6 Mestrado Profissional em Ciências do Meio Ambiente, Universidade Veiga de<br/>
 Almeida, 20271-901, Rio de Janeiro, Brazil.<br/>
7 Department of Ecology and Evolutionary Biology, University of Connecticut, 06269 Storrs, USA.<br/>
8 Tropical Forests and People Research Centre, University of the Sunshine Coast, 4558, Queensland, Australia.<br/>
9 Sustainable Farms, Fenner School of Environment and Society, The Australian National University, 2601, Canberra, Australia.<br/>
10 The Nature Conservancy, Brazil Program, 01311-936, São Paulo, Brazil.<br/>
11 Department of Forest Sciences, “Luiz de Queiroz” College of Agriculture, University of São Paulo, 13418-900, Piracicaba, Brazil.<br/>
<br/>
For more information please visit:<br/>
https://www.tnc.org.br/GoFor <br/>
www.nature.org<br/>
https://naturalcapitalproject.stanford.edu/ <br/>

            <br/>
            <b>Version {version}</b> <br/> <br/>
    
            Documentation: <a href="https://www.tnc.org.br/GoFor/">online</a><br/>
            Homepage: <a href="textwrap">
                        GoFor Home</a><br/>
            Copyright 2019, The Nature Conservancy<br/>
            License:
            <a href="https://bitbucket.org/natcap/invest/src/tip/LICENSE.txt">
                        BSD 3-clause</a><br/>
            Project page: <a href="https://www.tnc.org.br/GoFor/">
                        GoFor</a><br/>
    
            <h2>Open-Source Licenses</h2>
            """.format(
                version='1.0.0'))


        label_text += "<table>"
        for lib_name, lib_license, lib_homepage in [
            ('PyInstaller', 'GPL', 'http://pyinstaller.org'),
            ('GDAL', 'MIT and others', 'http://gdal.org'),
            ('matplotlib', 'BSD', 'http://matplotlib.org'),
            ('natcap.versioner', 'BSD',
             'http://bitbucket.org/jdouglass/versioner'),
            ('numpy', 'BSD', 'http://numpy.org'),
            ('pyamg', 'BSD', 'http://github.com/pyamg/pyamg'),
            ('pygeoprocessing', 'BSD',
             'http://bitbucket.org/richsharp/pygeoprocessing'),
            ('PyQt', 'GPL',
             'http://riverbankcomputing.com/software/pyqt/intro'),
            ('rtree', 'LGPL', 'http://toblerity.org/rtree/'),
            ('scipy', 'BSD', 'http://www.scipy.org/'),
            ('shapely', 'BSD', 'http://github.com/Toblerity/Shapely')]:
            label_text += (
                '<tr>'
                '<td>{project}  </td>'
                '<td>{license}  </td>'
                '<td>{homepage}  </td></tr/>').format(
                project=lib_name,
                license=(
                    '<a href="licenses/{project}_license.txt">'
                    '{license}</a>').format(project=lib_name,
                                            license=lib_license),
                homepage='<a href="{0}">{0}</a>'.format(lib_homepage))

        label_text += "</table>"
        label_text += textwrap.dedent(
            """
            <br/>
            <p>
            The source code for GPL'd components are included as an extra
            component on your <br/> installation medium.
            </p>
            """)

        self.about_dialog.label = QtWidgets.QLabel(label_text)
        self.about_dialog.label.setTextFormat(QtCore.Qt.RichText)
        self.about_dialog.label.setOpenExternalLinks(True)
        self.about_dialog.layout().addWidget(self.about_dialog.label)

        self.about_dialog.button_box = QtWidgets.QDialogButtonBox()
        self.about_dialog.accept_button = QtWidgets.QPushButton('OK')
        self.about_dialog.button_box.addButton(
            self.about_dialog.accept_button,
            QtWidgets.QDialogButtonBox.AcceptRole)
        self.about_dialog.layout().addWidget(self.about_dialog.button_box)
        self.about_dialog.accept_button.clicked.connect(self.about_dialog.close)

    def generate_args_from_inputs(self):
        """Used to geenrate args automatically rather than manuually adding them.
         e.g., args[self.create_simplied_lulc.args_key] = self.create_simplied_lulc.value(),
         Note that this then means that the args_key must be exactly correct."""
        args = OrderedDict()
        input_types_to_read = [
            inputs.Text,
            inputs.Checkbox,
            inputs.Container,
            inputs.Dropdown,
            inputs.File,
            inputs.FileButton,
            inputs.FolderButton,
            inputs.Folder,
            inputs.InVESTModelInput,
            inputs.Multi,
            inputs.FileWithDefault,
            inputs.TextWithDefault,
            # inputs.FileSystemRunDialog,
            # inputs.FileDialog,
        ]

        for k,v in self.__dict__.items():
            if type(v) in input_types_to_read:
                args[v.args_key] = v.value()
        return args

    def assemble_args(self):
        return self.generate_args_from_inputs()
