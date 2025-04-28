"""
Definition of custom QWidget to provide interface for Gemini module.
"""
from qtpy.QtCore import Qt, QSize, QItemSelectionModel
from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
    QGridLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QGroupBox,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QLineEdit,
    QFileDialog,
    QHBoxLayout,
    QLineEdit
)

from qtpy.QtGui import (
    QIcon,
    QPixmap,
    QColor,
    QImage
)

from napari.utils.notifications import (
    notification_manager,
    show_info,
)
from napari.experimental import link_layers
from napari.layers import Labels, Image
from napari.utils.colormaps import AVAILABLE_COLORMAPS, label_colormap
from napari.utils.colormaps.vendored import colors as color_utils
import numpy as np
import pandas as pd
from os import path, listdir, system, sep, name
import os
from sysconfig import get_path
from ntpath import sep
import subprocess

class GeminiQWidget(QWidget):
    def __init__(self, napari_viewer, gem):
        super().__init__()
        self.viewer = napari_viewer
        self.gem = gem

        vbox_layout = QVBoxLayout(self)
        vbox_layout.setContentsMargins(9, 9, 9, 9)

        self.setLayout(vbox_layout)

        self.createMorphologyImageWidget()
        if self.gem.is_protein:
            self.createProteinExpressionWidget()
        else:
            self.createTranscriptsWidget()
        self.createMetadataWidget()

        self.createStitchingWidget()

    def _on_morph_click(self):
        self.gem.add_channel(self.channelsComboBox.currentText(),
            self.channelsColormapComboBox.currentText())

    def _on_expr_click(self):
        self.gem.add_protein(self.proteinComboBox.currentText(), self.colormapComboBox.currentText())

    def _on_rna_click(self):
        self.gem.plot_transcripts(gene=self.targetsComboBox.currentText(),
            color=self.colorsComboBox.currentText())

    def _get_label_colors(self):
        if not self.showSelectedCheckbox.isChecked():
            colors = [i.icon().pixmap(QSize(1, 1)).toImage().pixelColor(0,0).name() for i in self.labelListWidget.items()]
            items = [i.text() for i in self.labelListWidget.items()]
            items = pd.Series(items).astype(self.gem.cells_layer.features[self.metaComboBox.currentText()].dtype) 
            return dict(zip(items, colors))
        else:
            colors = [i.icon().pixmap(QSize(1, 1)).toImage().pixelColor(0,0).name() for i in self.labelListWidget.selectedItems()]
            items = [i.text() for i in self.labelListWidget.selectedItems()]
            items = pd.Series(items).astype(self.gem.cells_layer.features[self.metaComboBox.currentText()].dtype) 
            return dict(zip(items, colors))

    def _labels_selected(self):
        if self.showSelectedCheckbox.isChecked():
            meta_col = self.metaComboBox.currentText()
            colors = self._get_label_colors()
            selected_items = [i.text() for i in self.labelListWidget.selectedItems()]
            selected_items = pd.Series(selected_items).astype(self.gem.cells_layer.features[meta_col].dtype) 
            cells = self.gem.cells_layer.features[self.gem.cells_layer.features[meta_col].isin(selected_items)]['index']
            self.gem.color_cells(meta_col, colors, subset=cells)
    
    def _show_selected_changed(self, state):
        if self.showSelectedCheckbox.isChecked():
            self._labels_selected()
        else:
            self._meta_changed(self.metaComboBox.currentText())

    def _meta_changed(self, text):
        self.showSelectedCheckbox.setChecked(False)
        if text == "" or text is None or not self.gem.is_categorical_metadata(text):
            self.labelListWidget.setHidden(True)
            self.showSelectedCheckbox.setHidden(True)
        else:
            self.updateLabelsWidget(text)
            self.labelListWidget.setHidden(False)
            self.showSelectedCheckbox.setHidden(False)
        if text != "" and text is not None:
            self.gem.color_cells(text)

    def _channel_changed(self, text):
        cmap = self.gem.omero(text, auto=False)['color']
        all_items = [self.channelsColormapComboBox.itemText(i)
            for i in range(self.channelsColormapComboBox.count())]
        if cmap in all_items:
            self.channelsColormapComboBox.setCurrentIndex(all_items.index(cmap))

    def _protein_changed(self, text):
        cmap = self.gem.omero(text, protein=True, auto=False)['color']
        all_items = [self.colormapComboBox.itemText(i)
            for i in range(self.colormapComboBox.count())]
        if cmap in all_items:
            self.colormapComboBox.setCurrentIndex(all_items.index(cmap))

    def _check_folder_validity(self) -> bool:
        """ Checks if a selected folder is a valid slide.

        Description: 
            A folder is a valid slide if it has the following folders:
            CellStatsDir, AnalysisResults/*, and RunSummary
        
        Returns:
            bool: True if valid, False if not valid
        """
        #print("in _check_folder_validity")
        isValid = True
        if not path.isdir(self.stitching_folder + '/CellStatsDir'):
            print("No valid CellStatsDir")
            isValid = False
        if not path.isdir(self.stitching_folder + '/RunSummary'):
            print("No valid RunSummary")
            isValid = False 
        if not path.isdir(self.stitching_folder + '/AnalysisResults'):
            print("No valid AnalysisResults Parent folder")
            isValid = False
        else: 
            # check if /AnalysisResults/<random_subfolder_name> exists
            analysis_sub_dir = [i for i in listdir(self.stitching_folder + '/AnalysisResults') if not i.startswith('.')]
            if(len(analysis_sub_dir)!=1):
                print("No valid AnalysisResults subfolder")
                isValid = False
        return isValid
    
    def _browse_folder(self):
        """ opens finder folder to select, 
            checks validity of folder, 
            and enables stitch button if valid.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if os.name == 'nt':
            print('Window detected! Change folder path from / to double (\\)')
            folder_path = folder_path.replace("/", r'\\')
        if folder_path:
            self.folder_path.setText(folder_path)
        msg = "The selected folder is" + str(folder_path)
        print(msg)
        self.stitching_folder = str(folder_path)
        isValidFolder = self._check_folder_validity()
        if isValidFolder:
            self.selected_folder_label.setText(f"Selected folder: {folder_path}")
            self.browser_output_button.setEnabled(True) # enable the stitch button
        else: 
            self.selected_folder_label.setText(f"Error: the selected folder, {folder_path}, is not a valid slide. Please select another folder.")
            self.browser_output_button.setEnabled(False) # disable the stitch button

    def _browse_output_folder(self):
        output_folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        print(output_folder_path)
        if os.name == 'nt':
            print('Window detected! Change folder path from / to double (\\)')
            output_folder_path = output_folder_path.replace("/", r'\\')
        if output_folder_path:
            self.folder_output_path.setText(output_folder_path)
        print(self.folder_output_path.text())
        self.selected_output_folder_label.setText(f"Result will be sent to: {output_folder_path}")
        self.stitch_button.setEnabled(True) # enable the stitch button

    def _stitch_images_in_widget(self):
        """ Lower-level function that does stitching.
        """
        self.stitch_button.setText("Running in the background...")
        msg = "Stitching started.\n"
        self.stitch_message_label.setText(msg)
        if os.name == 'nt':
            msg += 'Windows detected. Adjusting file names.\n'
            self.stitch_message_label.setText(msg)
            self.stitching_folder = self.stitching_folder.replace("/", "\\") 
        stitch_script_path = os.path.join(get_path("scripts"), "stitch-images")
        if os.name == 'nt':
            stitch_script_path += '.exe'
            stitch_script_path = stitch_script_path.replace("/", "\\")
        print(stitch_script_path)
        read_targets_script_path = os.path.join(get_path("scripts"), "read-targets")
        if os.name == 'nt':
            read_targets_script_path += '.exe'
            read_targets_script_path = read_targets_script_path.replace("/", "\\")
        if not path.isfile(stitch_script_path):
            self.stitch_message_label.setText(msg + "Could not find stitch-images in path " + stitch_script_path)
            print("could not fine stitch script in path: " + stitch_script_path)
            return None
        else:
            msg += "Found console script stitch-images.\n"
            self.stitch_message_label.setText(msg)
        if not path.isfile(read_targets_script_path):
            self.stitch_message_label.setText(msg + "Could not find read-targets")
        else:
            msg += "Found console script read-targets.\n"
            self.stitch_message_label.setText(msg)
        CellStatsDir = os.path.join(self.stitching_folder, "CellStatsDir")
        msg += "CellStatsDir is " + CellStatsDir + ".\n"
        self.stitch_message_label.setText(msg)
        RunSummaryDir = os.path.join(self.stitching_folder, 'RunSummary')
        msg += "RunSummaryDir is " + RunSummaryDir + ".\n"
        self.stitch_message_label.setText(msg)
        AnalysisDirParent = os.path.join(self.stitching_folder, 'AnalysisResults')
        AnalysisDirSubBasename = [i for i in listdir(AnalysisDirParent) if not i.startswith('.')]
        AnalysisDir = os.path.join(AnalysisDirParent, AnalysisDirSubBasename[0])
        msg += "AnalysisDir is " + AnalysisDir + ".\nRuning stitch-images...\n"
        self.stitch_message_label.setText(msg)
        cmd = [
            stitch_script_path, "-i", CellStatsDir, "-f", RunSummaryDir, "-o", self.folder_output_path.text()
        ]
        print(cmd)
        self._run_command(cmd)
        msg += "\nFinished stitching images.\nRunning read-targets...\n"
        self.stitch_message_label.setText(msg)
        cmd2 = [
            read_targets_script_path, AnalysisDir, "-o", self.folder_output_path.text()
        ]
        self.stitch_message_label.setText(msg)
        print(cmd2)
        self._run_command(cmd2)
        msg += "\nFinished reading targets!\nSee output folder for results.\n"
        self.stitch_message_label.setText(msg)
        self.stitch_button.setText("Reselect input/output to stitch again")
        self.stitch_button.setEnabled(False)

    def _run_command(self, command):
        subprocess.run(command)

    def update_metadata(self, path):
        # TODO: would be nice to merge metadata, but replace for now
        self.gem.read_metadata(path)
        if self.gem.metadata is not None:
            self.metaComboBox.clear()
            self.metaComboBox.addItems([i for i in self.gem.metadata.columns if i not in ['cell_ID', 'fov', 'CellId']])

    def createMorphologyImageWidget(self):
        groupBox = QGroupBox(self, title="Morphology Images")

        btn = QPushButton("Add layer")
        btn.clicked.connect(self._on_morph_click)

        boxp = QComboBox(groupBox)
        boxp.addItems(self.gem.channels)

        colormaps = [i for i in list(AVAILABLE_COLORMAPS.keys()) if i not in ['label_colormap', 'custom']]
        boxc = QComboBox(groupBox)
        boxc.addItems(colormaps)

        self.channelsComboBox = boxp
        self.channelsColormapComboBox = boxc

        vbox = QVBoxLayout(groupBox)
        grid = QGridLayout()

        grid.addWidget(QLabel('channel:'), 1, 0)
        grid.addWidget(self.channelsComboBox, 1, 1)
        grid.addWidget(QLabel('colormap:'), 2, 0)
        grid.addWidget(self.channelsColormapComboBox, 2, 1)
        vbox.addLayout(grid)
        vbox.addWidget(btn)
        groupBox.setLayout(vbox)

        self._channel_changed(self.channelsComboBox.currentText())
        self.channelsComboBox.currentTextChanged.connect(self._channel_changed)

        self.layout().addWidget(groupBox)

    def createProteinExpressionWidget(self):
        groupBox = QGroupBox(self, title="Protein Expression")

        btn = QPushButton("Add layer")
        btn.clicked.connect(self._on_expr_click)

        boxp = QComboBox(groupBox)
        boxp.addItems(self.gem.proteins)

        colormaps = [i for i in list(AVAILABLE_COLORMAPS.keys()) if i not in ['label_colormap', 'custom']]
        boxc = QComboBox(groupBox)
        boxc.addItems(colormaps)

        self.proteinComboBox = boxp
        self.colormapComboBox = boxc

        vbox = QVBoxLayout(groupBox)
        grid = QGridLayout()

        grid.addWidget(QLabel('protein:'), 1, 0)
        grid.addWidget(self.proteinComboBox, 1, 1)
        grid.addWidget(QLabel('colormap:'), 2, 0)
        grid.addWidget(self.colormapComboBox, 2, 1)
        vbox.addLayout(grid)
        vbox.addWidget(btn)
        groupBox.setLayout(vbox)

        self._protein_changed(self.proteinComboBox.currentText())
        self.proteinComboBox.currentTextChanged.connect(self._protein_changed)

        self.layout().addWidget(groupBox)

    def createTranscriptsWidget(self):
        groupBox = QGroupBox(self, title="RNA Transcripts")

        btn = QPushButton("Add layer")
        btn.clicked.connect(self._on_rna_click)

        boxp = QComboBox(groupBox)
        boxp.addItems(self.gem.genes)

        colors = ['white', 'red', 'green', 'blue',
            'magenta', 'yellow', 'cyan']
        boxc = QComboBox(groupBox)
        boxc.addItems(colors)

        self.targetsComboBox = boxp
        self.colorsComboBox = boxc

        vbox = QVBoxLayout(groupBox)
        grid = QGridLayout()

        grid.addWidget(QLabel('target:'), 1, 0)
        grid.addWidget(self.targetsComboBox, 1, 1)
        grid.addWidget(QLabel('color:'), 2, 0)
        grid.addWidget(self.colorsComboBox, 2, 1)
        vbox.addLayout(grid)
        vbox.addWidget(btn)
        groupBox.setLayout(vbox)

        self.layout().addWidget(groupBox)

    def createMetadataWidget(self):
        groupBox = QGroupBox(self, title="Color Cells")

        boxc = QComboBox(groupBox)
        boxc.toolTip = "Open a _metadata.csv file to populate dropdown"
        if self.gem.metadata is not None:
            boxc.addItems([i for i in self.gem.metadata.columns if i not in ['cell_ID', 'fov', 'CellId']])
        else:
            boxc.addItems([])

        self.metaComboBox = boxc

        vbox = QVBoxLayout(groupBox)
        grid = QGridLayout()

        grid.addWidget(QLabel('column:'), 1, 0)
        grid.addWidget(self.metaComboBox, 1, 1)
        vbox.addLayout(grid)

        listl = QListWidget(groupBox)
        listl.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.labelListWidget = listl
        self.showSelectedCheckbox = QCheckBox('Only show selected labels')
        self.showSelectedCheckbox.stateChanged.connect(self._show_selected_changed)

        self.labelListWidget.setHidden(True)
        self.showSelectedCheckbox.setHidden(True)
        self._meta_changed(self.metaComboBox.currentText())
        if self.gem.cells_layer is not None:
            self.gem.cells_layer.visible = False
        self.metaComboBox.currentTextChanged.connect(self._meta_changed)

        listl.itemSelectionChanged.connect(self._labels_selected)
        vbox.addWidget(self.labelListWidget)
        vbox.addWidget(self.showSelectedCheckbox)
        groupBox.setLayout(vbox)
        self.layout().addWidget(groupBox)

    def createStitchingWidget(self):
        # set initial value
        self.stitching_folder = "No folder selected." 
        self.folder_output_path = "No output folder selected."
        groupBox = QGroupBox(self, title="Stitch Images")

        # Widget description text
        folder_info_label = QLabel("To create napari-ready zarr files,\nselect raw data folder:")
        folder_info_label.setWordWrap(True)

        # Button for browser
        self.folder_path = QLineEdit()
        self.folder_path.setEnabled(False)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_folder)

        # Visual feedback for user (selected input path)
        self.selected_folder_label = QLabel(self.stitching_folder)
        self.selected_folder_label.setWordWrap(True)

        # Button for output folder
        self.folder_output_path = QLineEdit()
        self.folder_output_path.setEnabled(False)

        self.browser_output_button = QPushButton("Choose Output Folder")
        self.browser_output_button.clicked.connect(self._browse_output_folder)
        self.browser_output_button.setEnabled(False)

        # Visual feedback for user (selected output path)
        self.selected_output_folder_label = QLabel(self.folder_output_path)
        self.selected_output_folder_label.setWordWrap(True)

        # Button for calling stitching function
        self.stitch_button = QPushButton("Stitch")
        self.stitch_button.clicked.connect(self._stitch_images_in_widget)
        self.stitch_button.setEnabled(False)

        # Visual feedback for user (stitch path)
        self.stitch_message_label = QLabel("")
        self.stitch_message_label.setWordWrap(True)

        # Configure panels
        vbox = QVBoxLayout(groupBox)
        grid = QGridLayout()
        grid.addWidget(folder_info_label, 1, 0)
        vbox.addLayout(grid)
        vbox.addWidget(self.browse_button)
        vbox.addWidget(self.selected_folder_label)
        vbox.addWidget(self.browser_output_button)
        vbox.addWidget(self.selected_output_folder_label)
        vbox.addWidget(self.stitch_button)
        vbox.addWidget(self.stitch_message_label)
        groupBox.setLayout(vbox)    

        # render
        self.layout().addWidget(groupBox)

    def updateLabelsWidget(self, meta_col):
        self.labelListWidget.clear()
        vals = sorted(np.unique(self.gem.metadata[meta_col]))
        if self.gem.adata is not None and meta_col + "_colors" in self.gem.adata.uns:
            # get colors from AnnData object
            color = dict(zip(self.gem.adata.obs[meta_col].cat.categories, self.gem.adata.uns[meta_col + "_colors"]))
            cols = np.vstack((np.zeros(4, dtype='float64'),
                color_utils.to_rgba_array([color[i] for i in vals])))
        else:
            cols = label_colormap(len(vals)+1).colors
        for i,n in enumerate(vals):
            pmap = QPixmap(24, 24)
            rgba = cols[i+1]
            color = np.round(255 * rgba).astype(int)
            pmap.fill(QColor(*list(color)))
            icon = QIcon(pmap)
            qitem = QListWidgetItem(icon, str(n), self.labelListWidget)