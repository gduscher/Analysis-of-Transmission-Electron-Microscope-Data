##################################
#
# 2018 01 31 Included Nion Swift files to be opened
#
##################################

import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider

try:
    import h5py
except:
    print('h5py not installed, cannot read Nion files')
    h5py_installed = False

import pyUSID as usid
import json
import struct
import ipywidgets as ipyw
import ipywidgets as widgets
from IPython.display import display, clear_output
import sys, os

## Open/Save File dialog 
try:
    from PyQt5 import QtGui, QtWidgets
    QT_available = True
except:
    QT_available = False

try:
    import tkinter
    from tkinter import filedialog
    TK_available = True
except:
    TK_available = False

# =============================================================
#   Include Quantifit Libraries                                      #
# =============================================================

import pyTEMlib.dm3lib_v1_0b as dm3 # changed this dm3 reader library to support SI and EELS

from pyTEMlib.config_dir import config_path

####
#  General Open and Save Methods
####

def savefile_dialog(initial_file = '*.hf5', file_types = None):
    """
        Opens a save dialog in QT and retuns an "*.hf5" file. 
        New now with intial file
    """
    # Check whether QT is available
    if QT_available == False:
        print('No QT dialog')
        return None
    else:
        if file_types == None:
            file_types ="All files (*)"
    
    # Determine last path used
    try:
        fp = open(config_path+'\path.txt','r')
        path = fp.read()
        fp.close()
    except:
        path = ''
    
    if len(path)<2:
        path = '.'

    fname, file_filter = QtWidgets.QFileDialog.getSaveFileName(None, "Select a pyUSID file...", path+"/"+initial_file, filter=file_types)

    if len(fname) > 1:
        fp = open(config_path+'\path.txt','w')
        path, fileName = os.path.split(fname)
        fp.write(path)
        fp.close()

    if len(fname) > 3:
        h5_file_name = get_h5_filename(fname)
    else:
        return ''
def get_h5_filename(fname):
    path, filename = os.path.split(fname)
    basename, extension = os.path.splitext(filename)
    h5_file_name_original = os.path.join(path,basename+'.hf5')
    h5_file_name = h5_file_name_original
    
    if os.path.exists(os.path.abspath(h5_file_name_original)):
        count = 1
        h5_file_name = h5_file_name_original[:-4]+'-'+str(count)+'.hf5'
        while os.path.exists(os.path.abspath(h5_file_name)):
            count+=1
            h5_file_name = h5_file_name_original[:-4]+'-'+str(count)+'.hf5'
            
    if h5_file_name != h5_file_name_original:
        path, filename = os.path.split(h5_file_name)
        print('Cannot overwrite file. Using: ',filename)
    return str(h5_file_name)


def openfile_dialog(file_types = None, multiple_files = False, gui = 'None'):
    
    """
    Opens a File dialog which is used in open_file() function
    This functon uses tkinter or pyQt5.
    The app of the Gui has to be running for QT so Tkinter is a safer bet.
    In jupyter notebooks use %gui Qt early in the notebook.
    
    
    The file looks first for a path.txt file for the last directory you used.

    Parameters
    ----------
    file_types : string of the file type filter 
    gui: either 'Qt' or 'Tk' can be selected as Gui library for openfile dialog
    
    Returns
    -------
    filename : full filename with absolute path and extension as a string

    Examples
    --------
    
    >>> from config_dir import config_path
    >>> import file_tools as ft
    >>>
    >>> filename = ft.openfile_dialog()
    >>> 
    >>> print(filename)

    """
    if gui not in ['Qt','Tk']:
        if QT_available:
            gui = 'Qt'
        elif TK_available:
            gui = 'Tk'
        else:
            return None
    

    if file_types == None:
        file_types = 'TEM files (*.dm3 *.qf3 *.ndata *.h5 *.hf5);;pyUSID files (*.hf5);;QF files ( *.qf3);;DM files (*.dm3);;Nion files (*.ndata *.h5);;All files (*)'
    elif file_types == 'pyUSID':
        file_types = 'pyUSID files (*.hf5);;TEM files (*.dm3 *.qf3 *.ndata *.h5 *.hf5);;QF files ( *.qf3);;DM files (*.dm3);;Nion files (*.ndata *.h5);;All files (*)'
    
        #file_types = [("TEM files",["*.dm*","*.hf*","*.ndata" ]),("pyUSID files","*.hf5"),("DM files","*.dm*"),("Nion files",["*.h5","*.ndata"]),("all files","*.*")]
    # Determine last path used
    try:
        fp = open(config_path+'\path.txt','r')
        path = fp.read()
        fp.close()
    except:
        path = ''
    
    if len(path)<2:
        path = '.'

    if gui == 'Qt':
        if multiple_files:
            fnames, file_filter = QtWidgets.QFileDialog.getOpenFileNames(None, "Select a file...", path, filter=file_types)
            if len(fnames) >0:
                fname = fnames[0]
            else:
                return 
        else:
            fname, file_filter = QtWidgets.QFileDialog.getOpenFileName(None, "Select a file...", path, filter=file_types)
    
    else:
        multiple_files = False
        root = tkinter.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)

        tk_list = []
        for ffilter in  file_types.split(';;' ):
            split_filter = ffilter.split('(')
            tk_list.append((split_filter[0],split_filter[1][:-1].split(' ')))
        fnames =  filedialog.askopenfilename(initialdir = path, title = "Select file", filetypes = tk_list, multiple = multiple_files)
        
    if len(fname) > 1:
        
        fp = open(config_path+'\path.txt','w')
        path, fileName = os.path.split(fname)
        fp.write(path)
        fp.close()
    else:
        return ''
        
    if multiple_files:
         return fnames
    else:
        return str(fname)

def h5_get_dictionary(current_channel):
    tags = {}
    tags['aberrations'] = {}
    current_channel_tags = dict(current_channel.attrs)
    for key in current_channel_tags: ## Legacy Lines, original metadat should be in its own group
        if 'original' not in key:
            if 'aberration' in key:
                tags['aberrations'][key]=current_channel_tags[key]
            else:
                tags[key]=current_channel_tags[key]
    if 'title' in current_channel:
        tags['title'] = current_channel['title'][()]        
    tags['data_type'] = current_channel['data_type'][()]
    if tags['data_type']== 'EELS_spectrum':
        if 'Log_' in current_channel.name:
            pass
        else:
            tags['data'] = current_channel['Raw_Data'][0,:]
        tags['spectral_scale_x'] = current_channel['spectral_scale_x'][()]
        tags['spectral_units_x'] = current_channel['spectral_units_x'][()]
        tags['spectral_origin_x'] = current_channel['spectral_origin_x'][()]
        tags['spectral_size_x'] = float(current_channel['spectral_size_x'][()])
        tags['energy_scale'] = np.arange(tags['spectral_size_x'])*tags['spectral_scale_x']+tags['spectral_origin_x']
    elif tags['data_type']== 'image':
        if 'Log_' in current_channel.name:
            pass
        else:
            tags['data'] = np.reshape(current_channel['Raw_Data'][:, 0], (current_channel['spatial_size_x'][()],current_channel['spatial_size_y'][()]))
        tags['spatial_size_x'] = current_channel['spatial_size_x'][()]
        tags['spatial_size_y'] = current_channel['spatial_size_y'][()]
        tags['spatial_scale_x'] = current_channel['spatial_scale_x'][()]
        tags['spatial_scale_y'] = current_channel['spatial_scale_y'][()]
        tags['FOV_x'] = tags['spatial_scale_x'][()] * tags['spatial_size_x'][()]
        tags['FOV_y'] = tags['spatial_scale_y'][()] * tags['spatial_size_y'][()]
        tags['extent']=(0,tags['FOV_x'],tags['FOV_y'],0)
        tags['spatial_units']=current_channel['spatial_units'][()]
        
        
                        
    elif tags['data_type']== 'spectrum_image':
        if 'Log_' in current_channel.name:
            pass
        else:
            tags['cube'] = np.reshape(current_channel['Raw_Data'][:, :], (current_channel['spatial_size_x'][()],current_channel['spatial_size_y'][()],current_channel['spectral_size_x'][()]))
            tags['data'] = tags['cube'].sum(axis=2)
        tags['spatial_size_x'] = current_channel['spatial_size_x'][()]
        tags['spatial_size_y'] = current_channel['spatial_size_y'][()]
        tags['spatial_scale_x'] = current_channel['spatial_scale_x'][()]
        tags['spatial_scale_y'] = current_channel['spatial_scale_y'][()]
        tags['FOV_x'] = tags['spatial_scale_x'][()] * tags['spatial_size_x'][()]
        tags['FOV_y'] = tags['spatial_scale_y'][()] * tags['spatial_size_y'][()]
        tags['extent']=(0,tags['FOV_x'],tags['FOV_y'],0)
        if current_channel['spatial_units'][()] == '':
            tags['spatial_units']='nm'
        else:
            tags['spatial_units']=current_channel['spatial_units'][()]
        
        tags['spectral_scale_x'] = current_channel['spectral_scale_x'][()]
        tags['spectral_units_x'] = current_channel['spectral_units_x'][()]
        tags['spectral_origin_x'] = current_channel['spectral_origin_x'][()]
        tags['spectral_size_x'] = float(current_channel['spectral_size_x'][()])
        tags['energy_scale'] = np.arange(tags['spectral_size_x'])*tags['spectral_scale_x']+tags['spectral_origin_x']
    elif tags['data_type']== 'image_stack':
        tags['spatial_size_x'] = current_channel['spatial_size_x'][()]
        tags['spatial_size_y'] = current_channel['spatial_size_y'][()]
        tags['spatial_scale_x'] = current_channel['spatial_scale_x'][()]
        tags['spatial_scale_y'] = current_channel['spatial_scale_y'][()]
        image_size = current_channel['spatial_size_x'][()]*current_channel['spatial_size_y'][()]
        if 'Log_' in current_channel.name:
            pass
        else:
            tags['data'] = np.reshape(current_channel['Raw_Data'][:image_size, 0], (current_channel['spatial_size_x'][()],current_channel['spatial_size_y'][()]))
        if 'image_stack' in current_channel:
            tags['image_stack'] = np.array(current_channel['image_stack'][()])
        elif 'time_size' in current_channel:
            pass
        #[:, ], (current_channel['spatial_size_x'][()],current_channel['spatial_size_y'][()],current_channel['spatial_size_y'][()]))
        
    
        tags['FOV_x'] = tags['spatial_scale_x'][()] * tags['spatial_size_x'][()]
        tags['FOV_y'] = tags['spatial_scale_y'][()] * tags['spatial_size_y'][()]
        tags['extent']=(0,tags['FOV_x'],tags['FOV_y'],0)
        tags['spatial_units']=current_channel['spatial_units'][()]
    
    elif tags['data_type']== 'result':
        pass
    else:
        tags['data'] = np.zeros((1,1))
        if 'spatial_size_x' in current_channel:
            if 'spatial_size_y' in current_channel:

                if 'Log_' in current_channel.name:
                    pass
                else:
                    tags['data'] = np.reshape(current_channel['Raw_Data'][:, 0], (current_channel['spatial_size_x'][()],current_channel['spatial_size_y'][()]))
        
                tags['spatial_size_x'] = current_channel['spatial_size_x'][()]
                tags['spatial_size_y'] = current_channel['spatial_size_y'][()]
        
        if 'spatial_scale_x' in current_channel:
            tags['spatial_scale_x'] = current_channel['spatial_scale_x'][()]
        else:
            tags['spatial_scale_x'] = 1.
        if 'spatial_scale_y' in current_channel:
            tags['spatial_scale_y'] = current_channel['spatial_scale_y'][()]
        else:
            tags['spatial_scale_y'] = 1.

        tags['FOV_x'] = tags['spatial_scale_x'][()] * tags['spatial_size_x'][()]
        tags['FOV_y'] = tags['spatial_scale_y'][()] * tags['spatial_size_y'][()]
        tags['extent']=(0,tags['FOV_x'],tags['FOV_y'],0)
    
    if 'spatial_units' in current_channel:
        tags['spatial_units'] = current_channel['spatial_units'][()]
    else:
        tags['spatial_units'] = 'pixel'
    
    if tags['spatial_units'] in ['',' ']:
        tags['spatial_units'] = 'pixel'
    return tags
    
def h5_open_file(filename = None, saveFile = False):
    """
    Opens a file if the extension is .hf5, .dm3 or .dm4
    If no filename is provided the tk open_file windows opens

    Everything will be stored in a pyUSID style hf5 file.

    Subbroutines used:
        - dm_to_pyUSID
            - get_main_tags
            - get_additional tags
            - several pyUSID io fucntions
        -nion_to_pyUSID    

    """
    
    if filename == None:
        
        filename = openfile_dialog()
        if filename == '':
            return
    path, file_name = os.path.split(filename)
    basename, extension = os.path.splitext(file_name)

    
    if extension ==  '.hf5':
        h5_file = h5py.File(filename, 'a')

        dset = get_main_dataset(h5_file)
        current_channel = dset.parent
        if 'title' not in current_channel:
            current_channel['title'] = basename

        else:
            current_channel['title'][()] = basename
        dset.attrs['title'] = basename
        return h5_file
    else:
        tags = open_file(filename)  
        if saveFile: 
            h5_file_name = savefile_dialog(initial_file = tags['initial_file'])
        else:
            h5_file_name = get_h5_filename(tags['initial_file'])

        path, file_name = os.path.split(h5_file_name)
        basename, _ = os.path.splitext(file_name)

        
        if extension  in ['.dm3','.dm4','.ndata', '.h5']:
            h5_file =  h5py.File(h5_file_name, mode='a')

            current_channel = h5_file.create_group("Measurement_000/Channel_000")
            current_channel['title'] = basename

            dset = make_h5_dataset(current_channel, tags)
            dset.attrs['title'] = basename
            
            h5_file.flush()
            return h5_file
        else:
            print('file type not handled yet.')
            return

def open_file(file_name = None):


    #Open file
   
    if file_name == None:
        file_name = openfile_dialog('TEM files (*.dm3 *.qf3 *.ndata *.h5 *.hf5);;pyUSID files (*.hf5);;QF files ( *.qf3);;DM files (*.dm3);;Nion files (*.ndata *.h5);;All files (*)')

    tags = {}
    tags['filename']= file_name
    head, tail = os.path.split(file_name)
    tags['path'] = head
    path, tags['extension'] = os.path.splitext(file_name)
    tags['basename'] = tail[:-len(tags['extension'])]

    if tags['extension'] == '.ndata':
        open_nion_file(file_name,tags)
        if 'file_type' not in tags:
            tags['file_type'] = 'image'
        tags['initial_file'] = tags['original_metadata']['title']+'.hf5'
    elif tags['extension'] == '.h5':
        open_h5_nion_file(file_name,tags)
        tags['initial_file'] = tags['original_metadata']['title']+'.hf5'
        
    elif tags['extension'] == '.dm3':
        open_dm3_file(file_name,tags)
        tags['initial_file'] = tags['basename']+'.hf5'
    elif tags['extension'] == '.qf3':
        open_qf3_file(file_name,tags)
        if 'data_type' not in tags:
            
            if len(tags['ene'])> 12:
                tags['data_type'] = 'EELS'
            else:
                tags['data_type'] = 'Image'

        

    else:
        tags['filename'] == 'None'
        print('io 1 no')

    
  
    tags['filename']= file_name
    head, tail = os.path.split(file_name)
    tags['path'] = head
    path, tags['extension'] = os.path.splitext(file_name)
    tags['basename'] = tail[:-len(tags['extension'])]

    #plot_tags(tags, 'new')
    return tags

##
# Loging of Information, Results and correlated data
##

def h5_log_calculation(h5_file,current_channel,tags):
    measurement_group = h5_file[current_channel.name.split('/')[1]]
    i = 0
    for key in measurement_group:
        if 'Calculation'in key:
            i+=1
    name = f'Calculation_{i:03d}'
    if 'data' not in tags:
        print('no data to log')
        return
    
    calc_grp = measurement_group.create_group(name)
    calc_grp['time_stamp']= usid.io.io_utils .get_time_stamp()
    for key in tags:
        calc_grp[key]= tags[key]

    h5_file.flush()
    return calc_grp

def h5_add_measurement(h5_file,current_channel,title):
    new_measurement_group = usid.io.hdf_utils.create_indexed_group(h5_file,'Measurement')
    
    return new_measurement_group

def h5_add_channels(h5_file,current_channel,title):
    file_filters = 'TEM files (*.dm3 *.qf3 *.ndata *.h5 *.hf5);;pyUSID files (*.hf5);;QF files ( *.qf3);;DM files (*.dm3);;Nion files (*.ndata *.h5);;All files (*)'
    filenames = openfile_dialog(file_filters, multiple_files=True)
    if filenames== None:
        print('File Selection canceled')
        return current_channel
    if len(filenames) == 0:
        return current_channel
    for file in filenames:
        current_channel = h5_add_channel(h5_file,current_channel,title,filename = file)
    return current_channel    

def h5_add_channel(h5_file,current_channel,title,filename=None):

    #Open file
   
    if filename == None:
        file_filters = 'TEM files (*.dm3 *.qf3 *.ndata *.h5 *.hf5);;pyUSID files (*.hf5);;QF files ( *.qf3);;DM files (*.dm3);;Nion files (*.ndata *.h5);;All files (*)'
        filename = openfile_dialog(file_filters)
        if filename== None:
            print('File Selection canceled')
            return
        if filename == '':
            return
    path, file_name = os.path.split(filename)
    basename, extension = os.path.splitext(file_name)
    time_last_modified = os.path.getmtime(filename)
    
    #if extension ==  '.hf5':
    #    return h5py.File(filename, 'a')
    if extension !=  '.dm3':
        print('file type is not handled yet')
    
    ### 
    # Read DM file and extract the main information
    ###
    
    si = dm3.DM3(filename)
    main_tags, channel_tags, data_tags  = get_main_tags(si)
    
    ###
    # Open pyUSID file and write data and main meta-data
    ###
    
    tran = usid.NumpyTranslator()
    h5_file_name = filename[:-4]+'.hf5'
    
    quantity = 'distance'
    units = 'nm'
    if current_channel == None:
        return
        
    else:
        '''
        measurement_group = h5_file[current_channel.name.split('/')[1]]
        i = 0
        for key in measurement_group:
            if 'Channel'in key:
                i+=1
        name = f'Channel_{i:03d}'
        if 'rawData' not in data_tags:
            print('no data to add')
            return
        '''
        measurement_group = h5_file[current_channel.name.split('/')[1]]
        name = usid.io.hdf_utils.assign_group_index(measurement_group,'Channel')
        
        
        current_channel = measurement_group.create_group(name)
        _ = usid.io.hdf_utils.write_main_dataset(current_channel, # parent HDF5 group
                                                 data_tags['rawData'], 
                                                 'Raw_Data', # Name of main dataset
                                                 'distance', 
                                                 'nm',  
                                                 data_tags['pos_dims'], 
                                                 data_tags['spec_dims'])
    
    current_channel.create_dataset('title',data = title)
    current_channel.create_dataset('filename', data = basename)
    current_channel.create_dataset('data_type', data = data_tags['data_type'])
    for key in channel_tags:
        current_channel.create_dataset(key, data=channel_tags[key])
    
    ###
    # Read Additional Meta_Data
    ###
    channel_tags['data_type'] = data_tags['data_type']
    meta_tags = get_additional_tags(si,channel_tags)
    meta_tags['time_last_modified'] = time_last_modified
    
	###
	# Write additional important metadata and original_metadata to current_channel attributes
    ###
    current_channel_tags = current_channel.attrs
    for key in meta_tags:
        #print(key,meta_tags[key])
        if 'DM' in key:
            pass
        else:
            current_channel_tags[key]= meta_tags[key]
    
    h5_file.flush()
    return current_channel

def log_results(current_channel, info_dictionary):
    
    log_group = h5_add_Log(current_channel, info_dictionary['name'])
    
    if 'data' in info_dictionary:
        dset = log_group.create_dataset("nDim_Data", data=info_dictionary['data'])
        #dset = log_group['nDim_Data']
        make_dimensions(dset, info_dictionary)
        dset.attrs['title'] = info_dictionary['title']
    
    dimension_labels = ['data', 'cube','data_type','title','spectra_size_x', 'spectra_offset_x', 'spectra_inits_x', 'spectra_scale_x','spatial_scale_x', 'spatial_origin_x', 'spatial_scale_y', 'spatial_origin_x', 'spatial_origin_y',  'spatial_size_x', 'spatial_size_y', 'spatial_units']
    
    for key in info_dictionary:
        if key not in dimension_labels:
            log_group[key] = info_dictionary[key]
    
    return log_group


def h5_add_Data2Log(log_group, info_dictionary):
    for key in info_dictionary:
        log_group[key] = info_dictionary[key]

def h5_add_Log(current_channel, name):
    log_group = usid.io.hdf_utils.create_indexed_group(current_channel,'Log')
    log_group['title'] = name
    log_group['_'+name] = name ## just easier to read the file-tree that way 
    log_group['time_stamp']= usid.io.io_utils.get_time_stamp()
    try:
        log_group['notebook'] = __notebook__
        log_group['notebook_version'] = __notebook_version__
    except:
        pass
    return log_group

class choose_image(object):
    
    def __init__(self, current_channel):
        
        self.current_channel = current_channel
        
        self.get_images()
        
        
        self.select_image = widgets.Dropdown(
                                        options=self.image_names,
                                        value=self.image_names[0],
                                        description='Select image:',
                                        disabled=False,
                                        button_style=''
                                    )
        display(self.select_image)
        self.select_image.observe(self.set_image_channel, names='value')
        self.select_image.index = (len(self.image_names)-1)
    
    def get_images(self, directory = None):

        choices_of_images = []
        self.image_names = ['Original Data']
        self.image_channels = ['self.current_channel']
        self.log_keys = ['None']

        image_analysis = ['Registration', 'Lucy_Richardson', 'Undistorted', 'crop image']

        for key in self.current_channel:
            if 'Log' in key:
                if 'analysis' in self.current_channel[key]:
                    for name in image_analysis:
                        if name in self.current_channel[key]['analysis'][()]:
                            self.image_names.append(self.current_channel[key]['analysis'][()])
                            self.image_channels.append(self.current_channel[key])
                            self.log_keys.append(key)
                    
                else:
                    print(key, dict(self.current_channel[key]).keys())

    def set_image_channel(self,b):
        
        index = self.select_image.index
        self.image_channel = self.image_channels[index]
        self.data_set = self.image_channel['nDim_Data']
        self.log_key = self.log_keys[index]

##
# Convert old h5_file format 
# ##
def get_main_dataset(h5_file):
    h5_dataset = None
    
    current_channel = get_main_channel(h5_file)
    
    if 'Raw_Data' in current_channel:
        h5_dataset = convert_to_ndim(current_channel)
    elif 'nDim_Data' in current_channel:
        h5_dataset = current_channel['nDim_Data']
            
    h5_file.flush()
    
    return h5_dataset

def get_main_channel(h5_file):
    current_channel = None
    if 'Measurement_000' in h5_file:
        if 'Measurement_000/Channel_000' in h5_file:
            current_channel = h5_file['Measurement_000/Channel_000']
            
    return current_channel

def convert_to_ndim(current_channel):
    """
    make a n-dimensional hdf5 dataset from pyUSID 
    """
    
    # dset = current_channel['Raw_Data']
    tags = h5_get_dictionary(current_channel)
    
    if 'image_stack' in current_channel:
        if 'nDim_Data' not in current_channel:
            dset_temp = current_channel['image_stack']
            current_channel['nDim_Data'] = current_channel['image_stack'] ## hard link : same data new additional name
    elif 'data' in current_channel:
        if 'nDim_Data' not in current_channel:
            dset_temp = current_channel['data']
            current_channel['nDim_Data'] = current_channel['data'] ## hard link : same data new additional name
            
    elif 'Raw_Data' not in current_channel:
        return
    
    else:
        data = tags['data']
        if 'nDim_Data' not in current_channel:
            dset = current_channel.create_dataset("nDim_Data", data=data)
        else:
            print('why are you here?')
    
    dset = current_channel["nDim_Data"] 
    make_dimensions(dset, tags)
    #current_channel['x_axis'] = np.linspace(data.shape[0])*current_channel['spatial_scale_x']
    
    if 'image' in current_channel['data_type'][()]:
        FOV_x = current_channel['x_axis'][-1] - current_channel['x_axis'][0]
        FOV_y = current_channel['y_axis'][-1] - current_channel['y_axis'][0]
        dset.attrs['image_extent'] = [0,FOV_x,FOV_y,0]
    
    dset.attrs['data_type'] = current_channel['data_type'][()]
    if 'image_mode' in current_channel:
        dset.attrs['data_mode'] = current_channel['image_mode'][()]

    if 'acceleration_voltage' in current_channel:
        dset.attrs['acceleration_voltage'] = current_channel['acceleration_voltage'][()]

    
    return dset
    
def make_dimensions(dset, tags):
    current_channel = dset.parent
 
    if 'x_axis' in current_channel:
        del current_channel['x_axis']
        
    ## make dimension of dataset
   
    if 'spatial_scale_x' in tags:
        dset.dims[0].label = 'x'
        current_channel['x_axis'] = np.arange(dset.shape[0])*tags['spatial_scale_x']
        if 'spatial_offset_x' in tags:
            current_channel['x_axis'][()] = current_channel['x_axis'][()] + tags['spatial_offset_x']
        
        if 'spatial_unit_x' in tags:
            units = tags['spatial_unit_x']
        elif 'spatial_units' in tags:
            units = tags['spatial_units']
        else:
            units = ' '

        current_channel['x_axis'].make_scale(f"distance [{units}]")
        dset.dims[0].attach_scale(current_channel['x_axis'])
        dset.attrs['x_units'] = units
        
        
    if 'y_axis' in current_channel:
        del current_channel['y_axis']
        
    if 'spatial_scale_y' in tags:
        dset.dims[1].label = 'y'
        current_channel['y_axis'] = np.arange(dset.shape[1])*tags['spatial_scale_y']
        if 'spatial_offset_y' in tags:
            current_channel['y_axis'][()] = current_channel['y_axis'][()] + tags['spatial_offset_y']
        
        if 'spatial_unit_y' in tags:
            units = tags['spatial_unit_y']
        elif 'spatial_units' in tags:
            units = tags['spatial_units']
        else:
            units = ' '

        current_channel['y_axis'].make_scale(f"distance [{units}]")
        dset.dims[1].attach_scale(current_channel['x_axis'])
        dset.attrs['y_units'] = units
    
    if 'z_axis' in current_channel:
        del current_channel['z_axis']
        
    if 'image_stack' == tags['data_type']:
        dset.dims[2].label = 'z'
        current_channel['z_axis'] = np.arange(dset.shape[2])
        current_channel['z_axis'].make_scale(f"frame")
        dset.dims[2].attach_scale(current_channel['z_axis'])
        dset.attrs['z_units'] = 'frame'
        
    if 'spectral_size_x'in tags:
        ndim = len(dset.shape)-1
        dset.dims[ndim].label = 'energy_loss'

        if 'spectral_unit_x' in tags:
            units = tags['spectral_unit_x']
        elif 'spectral_units' in tags:
            units = tags['spectral_units']
        else:
            units = ' '
        
        current_channel['energy_axis'] = np.arange(dset.shape[ndim])*tags['spectral_scale_x']+tags['spectral_offset_x']
        current_channel['energy_axis'].make_scale(f"energy-loss [{units}]")
        dset.dims[ndim].attach_scale(current_channel['energy_axis'])
        dset.attrs['spectral_units'] =units
    if 'data_type' in current_channel:
        current_channel['data_type'][()] = tags['data_type']    
    else:
        current_channel['data_type'] = tags['data_type']
    dset.attrs['data_type'] = tags['data_type']

    
def clean_Log(current_channel):
    if 'nDim_Data' in current_channel:
        dset = current_channel['nDim_Data']
    else: 
        dset = convert_to_ndim(current_channel)
    if 'Raw_Data' in current_channel:
        del current_channel['Raw_Data']
    if 'image_stack' in current_channel:
        if 'data' in current_channel:
            del current_channel['data']
    return dset    
##
# Convert original data
##

def get_dimension_tags(tags):
    tags_dimension = {}
    
    ### scan axis dictionary
    stack_dimension = -1
    eels_dimension  = -1
    image_dimensions = []
    for dim, axis in tags['axis'].items():
        if axis['units'] == '':
            if len(tags['data'].shape) ==3:
                stack_dimension = int(dim)
        elif axis['units'] == 'eV':
            eels_dimension = int(dim)
            tags_dimension['spectral_scale_x'] = axis['scale']
            tags_dimension['spectral_size_x'] = tags['data'].shape[eels_dimension]
            tags_dimension['spectral_offset_x'] = axis['offset']
            tags_dimension['spectral_unit_x'] = axis['units']
        elif axis['units'] == 'nm':
            image_dimensions.append(int(dim))
            if len (image_dimensions) == 1:
                tags_dimension['spatial_scale_x'] = axis['scale']
                tags_dimension['spatial_size_x'] = tags['data'].shape[0]
                tags_dimension['spatial_offset_x'] = axis['offset']
                tags_dimension['spatial_unit_x'] = axis['units']
            elif len (image_dimensions) == 2:
                tags_dimension['spatial_scale_y'] = axis['scale']
                tags_dimension['spatial_size_y'] = tags['data'].shape[0]
                tags_dimension['spatial_offset_y'] = axis['offset']
                tags_dimension['spatial_unit_y'] = axis['units']
            
    ## get data and  data_type
    if 'cube' in tags:
        tags_dimension['data'] = tags['cube']
    else:
        tags_dimension['data'] = tags['data']

    

    if eels_dimension >-1:
        if len(image_dimensions) == 0:
            tags_dimension['data_type'] = 'eels_spectrum'
        elif len(image_dimensions) == 1:
            tags_dimension['data_type'] = 'eels_linescan'
        elif len(image_dimensions) == 2:
            tags_dimension['data_type'] = 'spectrum_image'
            if eels_dimension !=2:
                tags_dimension['data'] = np.swapaxes(tags['data'],stack_dimension, 2)

    if len(image_dimensions) == 2:
        tags_dimension['data_type'] = 'image'
    if stack_dimension >-1:
        if stack_dimension !=2:
            tags_dimension['data'] = np.swapaxes(tags['data'],stack_dimension, 2)
        tags_dimension['data_type'] = 'image_stack'
    
    if 'data_type' not in tags_dimension:
        tags_dimension['data_type'] = tags['data_type']
    
    return tags_dimension

def make_h5_dataset(current_channel, tags):
    
    tags_dimension = get_dimension_tags(tags)
    
    if "nDim_Data" in current_channel:
        dset = current_channel["nDim_Data"]
    else:
        dset = current_channel.create_dataset("nDim_Data", data=tags_dimension['data'])
    
    make_dimensions(dset, tags_dimension)
    
    
    def flatten(d, parent_key='', sep='.'):
            items = []

            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    if new_key == 'dimensional_calibrations':
                        for i in range(len(v)):
                            for kk in v[i]:
                                items.append(('dim-'+kk+'-'+str(i), v[i][kk]))
                    else:
                        items.append((new_key, v))

            return dict(items)

    original_group = current_channel.create_group('original_metadata')
    original_group_tags = original_group.attrs

    original_tags = flatten(tags['original_metadata'])


    if 'aberrations' in tags:
        aberrations_group = current_channel.create_group('aberrations')
        aberrations_group_tags = original_group.attrs
        for key, value in tags['aberrations'].items():
            aberrations_group_tags[key]= value

    if 'image' in tags:
        for key, value in tags['image'].items():
            dset.attrs[key]= value
    for key, value in tags.items():
        if isinstance(value,dict):
            pass
        elif 'data' == key:
            pass
        else:
            if key not in ['cube', 'spectrum']:
                dset.attrs[key]= value
            
    return dset


###
# Nion specific
###

def open_h5_nion_file(file_name,tags):
    fp = h5py.File(file_name, 'a')
    if 'data' in fp:
        json_properties = fp['data'].attrs.get("properties", "")
        data = fp['data'][:]
        tags['shape'] = data.shape
        dic = json.loads(json_properties)
        tags['data'] = data
        tags['original_metadata'] = dic
        tags['file_type'] = 'ndata'
        tags.update(get_nion_tags(dic,data))

        return tags

def get_nion_tags(dic,data):

    tags={}
    tags['data'] = data
    if 'description' in dic:
        tags['original_name'] = dic['description']['title']
    else:
        if 'title' in dic:
            tags['original_name'] = dic['title']
    if 'created' in dic:
        tags['created_date'] = dic['created'][0:10]
        tags['created_time'] = dic['created'][11:19]

    if 'data_source' in dic:
         dic = dic['data_source']
    
    if 'data_shape' in dic:
        tags['shape'] = dic['data_shape']

    spectrum = -1
    if 'dimensional_calibrations' in dic:
        tags['axis'] = {}
        for i in range(len(dic['dimensional_calibrations'])):
            tags['axis'][str(i)]= {}
            tags['axis'][str(i)]['offset'] = dic['dimensional_calibrations'][i]['offset']
            tags['axis'][str(i)]['scale']  = dic['dimensional_calibrations'][i]['scale']
            tags['axis'][str(i)]['units']  = dic['dimensional_calibrations'][i]['units']
            tags['axis'][str(i)]['pixels']  = tags['shape'][i]
            if tags['axis'][str(i)]['units'] == 'nm':
                tags['axis'][str(i)]['offset'] = 0.0
            if tags['axis'][str(i)]['units'] == 'eV':
                spectrum = i
        if spectrum > 0:
            tags['EELS_dimension'] = spectrum
            tags['dispersion'] = tags['axis'][str(spectrum)]['scale']
            tags['offset'] = tags['axis'][str(spectrum)]['offset']
            if len(tags['shape'])==1:
                tags['data_type'] = 'EELS_spectrum'
                tags['spec'] = data
            elif len(tags['shape'])==2:
                tags['data_type'] = 'EELS_linescan'
                tags['spec'] = data[0,:]
                tags['cube'] = data
                tags['SI'] = {}
                tags['SI']['data'] = data
            elif len(tags['shape'])==3:
                tags['data_type'] = 'spectrum_image'
                tags['spec'] = data[0,0,:]
                tags['SI'] = {}
                tags['SI']['data'] = data
                tags['cf.ndaube'] = data
        else:    
            if '1' in  tags['axis']:
                if tags['axis']['0']['units'] == tags['axis']['1']['units']:
                    tags['data_type'] = 'image'
        tags['pixel_size'] = tags['axis']['0']['scale']
        tags['FOV'] = tags['shape'][0] * tags['pixel_size']
    if 'metadata' in  dic:
        if 'hardware_source' in dic['metadata']:
            hs = dic['metadata']['hardware_source']
    
            tags['pixel_size'] = tags['axis']['0']['scale']
    
            if 'fov_nm' in hs: 
                tags['FOV'] = hs['fov_nm']
                tags['exposure'] = hs['exposure']
                tags['pixel_time_us'] = hs['pixel_time_us']
                tags['unit'] = 'nm'
                tags['pixel_size'] = tags['FOV']/tags['shape'][0]
                tags['acceleration_voltage'] = hs['autostem']['ImageScanned:EHT']
                if tags['acceleration_voltage'] > 100000.:
                    tags['microscope'] = 'UltraSTEM200'
                else:
                    tags['microscope'] = 'UltraSTEM100'
                tags['type'] = hs['channel_name']
                tags['aberrations'] = {}
                tags['image']={}
                tags['image']['exposure'] = hs['exposure']
                tags['image']['pixel_time_us'] = hs['pixel_time_us']
                tags['image']['ac_line_sync'] = hs['ac_line_sync']
                tags['image']['rotation_deg'] = hs['rotation_deg']
                
                for key2 in hs['autostem']:
                    if 'ImageScanned' in key2:
                         ## For consistency we remove the . in the aberrations 
                         name = key2[13:].replace(".", "")
                         if name[0] == 'C':
                             tags['aberrations'][name] =  hs['autostem'][key2]*1e9 # aberrations in nm
                         elif name[0:4] == 'BP2^':
                                 tags['image'][name[4:]] = hs['autostem'][key2]

                         else: 
                            tags['image'][name] =  hs['autostem'][key2]

                                    
    assume_corrected = ['C10','C12a','C12b']
    if 'aberrations' in tags:
        for key2 in assume_corrected:
            tags['aberrations'][key2] = 0.
    if 'image' in tags:
        if tags['image']['EHT'] > 101000:
            tags['aberrations']['source_size'] = 0.051
        elif tags['image']['EHT'] < 99000:
            tags['aberrations']['source_size'] = 0.081
        else:
            tags['aberrations']['source_size'] = 0.061

    return tags

def open_nion_file(file_name,tags):
    fp = open(file_name, "rb")
    local_files, dir_files, eocd = parse_zip(fp)
    
    contains_data = b"data.npy" in dir_files
    contains_metadata = b"metadata.json" in dir_files
    file_count = contains_data + contains_metadata # use fact that True is 1, False is 0
    
    fp.seek(local_files[dir_files[ b"data.npy"][1]][1])
    
    tags['data'] = np.load(fp)
    tags['shape'] = tags['data'].shape
    
        
    json_pos = local_files[dir_files[b"metadata.json"][1]][1]
    json_len = local_files[dir_files[b"metadata.json"][1]][2]
    fp.seek(json_pos)
    json_properties = fp.read(json_len)
    fp.close()
    dic = json.loads(json_properties.decode("utf-8"))
    if 'dimensional_calibrations' in dic:
        for dim in dic['dimensional_calibrations']:
            if dim['units'] == '':
                dim['units'] = 'pixels'

    

    tags['original_metadata'] = dic
    
    tags.update(get_nion_tags(dic,tags['data']))
    
    return tags

def parse_zip(fp):
    
    """
        Parse the zip file headers at fp
        :param fp: the file pointer from which to parse the zip file
        :return: A tuple of local files, directory headers, and end of central directory
        The local files are dictionary where the keys are the local file offset and the
        values are each a tuple consisting of the name, data position, data length, and crc32.
        The directory headers are a dictionary where the keys are the names of the files
        and the values are a tuple consisting of the directory header position, and the
        associated local file position.
        The end of central directory is a tuple consisting of the location of the end of
        central directory header and the location of the first directory header.
        This method will seek to location 0 of fp and leave fp at end of file.

        This function is copied from  nionswift/nion/swift/model/NDataHandler.py

    """
    local_files = {}
    dir_files = {}
    eocd = None
    fp.seek(0)
    while True:
        pos = fp.tell()
        signature = struct.unpack('I', fp.read(4))[0]
        if signature == 0x04034b50:
            fp.seek(pos + 14)
            crc32 = struct.unpack('I', fp.read(4))[0]
            fp.seek(pos + 18)
            data_len = struct.unpack('I', fp.read(4))[0]
            fp.seek(pos + 26)
            name_len = struct.unpack('H', fp.read(2))[0]
            extra_len = struct.unpack('H', fp.read(2))[0]
            name_bytes = fp.read(name_len)
            fp.seek(extra_len, os.SEEK_CUR)
            data_pos = fp.tell()
            fp.seek(data_len, os.SEEK_CUR)
            local_files[pos] = (name_bytes, data_pos, data_len, crc32)
        elif signature == 0x02014b50:
            fp.seek(pos + 28)
            name_len = struct.unpack('H', fp.read(2))[0]
            extra_len = struct.unpack('H', fp.read(2))[0]
            comment_len = struct.unpack('H', fp.read(2))[0]
            fp.seek(pos + 42)
            pos2 = struct.unpack('I', fp.read(4))[0]
            name_bytes = fp.read(name_len)
            fp.seek(pos + 46 + name_len + extra_len + comment_len)
            dir_files[name_bytes] = (pos, pos2)
        elif signature == 0x06054b50:
            fp.seek(pos + 16)
            pos2 = struct.unpack('I', fp.read(4))[0]
            eocd = (pos, pos2)
            break
        else:
            raise IOError()
    return local_files, dir_files, eocd

class nion_directory(object):
    
    def __init__(self, dir_name = None, extension = ['*']):
        
        
        if dir_name == None:
            self.get_directory()
        elif dir_name == '.':
            fp = open(config_path+'\path.txt','r')
            path = fp.read()
            fp.close()
            self.get_directory(path)
            self.dir_name = path
        elif os.path.isdir(dir_name):
            self.get_directory(dir_name)
            self.dir_name = dir_name
        else:
            self.dir_name = '.'
            self.get_directory(self.dir_name)

        self.dir_list==['.']
        self.extensions = extension
        
        self.select_Nion_files = widgets.Select(
                                        options=self.dir_list,
                                        value=self.dir_list[0],
                                        description='Select file:',
                                        disabled=False,
                                        rows=10,
                                        layout=widgets.Layout(width='70%')
                                    )
        display(self.select_Nion_files)
        self.set_options()
        self.select_Nion_files.observe(self.get_and_open, names='value')
    
    def get_directory(self, directory = None):
        if directory == None:
            directory = str(QtWidgets.QFileDialog.getExistingDirectory( None, "Select Directory"))
        
        self.dir_name = directory
        self.dir_dictionary ={}
        self.dir_list = []
        i = 0
        self.dir_list =  ['.','..']+os.listdir(directory)

    def set_options(self):
        self.dir_name = os.path.abspath(os.path.join(self.dir_name, self.dir_list[self.select_Nion_files.index]))
        dir_list = os.listdir(self.dir_name)
        
        file_list =[]
        display_file_list = []
        directory_list = []
        

        for i in range(len(dir_list)):
            name = dir_list[i]
            full_name= os.path.join(self.dir_name,name)
            
            if os.path.isfile(full_name):
                size = os.path.getsize(full_name)*2**-20
                basename, extension = os.path.splitext(name)
                if self.extensions[0] == 'hf5':
                    if  extension in ['.hf5']:
                        file_list.append(dir_list[i])
                        display_file_list.append(f" {name}  - {size:.1f} MB")
                else:    
                    file_list.append(dir_list[i])
                    if extension in ['.h5', '.ndata']:
                        tags = open_file(os.path.join(self.dir_name,name))
                        
                        display_file_list.append(f" {tags['original_name']}{extension}  - {size:.1f} MB")
                    elif extension in ['.hf5']:
                        display_file_list.append(f" {name}  -- {size:.1f} MB")
                    else:
                        display_file_list.append(f' {name}  - {size:.1f} MB')
            else:
                directory_list.append(name)

        sort = np.argsort(directory_list)
        self.dir_list = ['.','..']
        self.display_list = ['.','..']
        for j in sort:
            self.display_list.append(f' * {directory_list[j]}')
            self.dir_list.append(directory_list[j])

        sort = np.argsort(display_file_list)
        
        for i , j in enumerate(sort):
            if '--' in dir_list[j]:
                self.display_list.append(f' {i:3} {display_file_list[j]}')
            else:
                self.display_list.append(f' {i:3}   {display_file_list[j]}')
            self.dir_list.append(file_list[j])
            
        self.dir_label = os.path.split(self.dir_name)[-1]+':'
        self.select_Nion_files.options = self.display_list


    def get_and_open(self,b):
        #global h5_file, current_channel
        #clear_output
        #print(select_Nion_files.value, dir_dictionary[select_Nion_files.value])
        
        
        if os.path.isdir(os.path.join(self.dir_name, self.dir_list[self.select_Nion_files.index])):
             self.set_options()
            
            
        elif  os.path.isfile(os.path.join(self.dir_name, self.dir_list[self.select_Nion_files.index])):
            
            file_name = os.path.join(self.dir_name, self.dir_list[self.select_Nion_files.index])      
            
            try:
                self.h5_file.close()
            except:
                pass


            self.h5_file = h5_open_file(file_name, saveFile = False)
            self.current_channel = self.h5_file['Measurement_000/Channel_000']
            print('loaded: ', self.current_channel['title'][()],' - ',self.dir_list[self.select_Nion_files.index] )

##
# DM3 specific
##

def getDictionary(intags):
    
    tags = {}
    tags['type']='DM'
    tags['file version'] = 3
    
    for line in intags.keys():
        if '.ImageList.1.' in line:
            keys = line[17:].split('.')
            tags_before = tags
            for i in range(len(keys)-1):
                if keys[i] not in tags_before:
                    tags_before[keys[i]] = {}
                tags_before = tags_before[keys[i]]
            
            tags_before[keys[-1]] = intags[line]
        if 'AnnotationGroupList' in line:
            if 'Label' in line:            
                if  self.__tagDict[line] == 'Spectrum Image':
                     tags['spectrum_image'] ={}
                     base = line[0:-5]
                     tags['spectrum_image']['rectangle'] = self.__tagDict[base+'Rectangle']
            
    return tags

def open_dm3_file(file_name,tags):
    
    si = dm3.DM3(file_name)
    data = si.data_cube
    dm = getDictionary(si.getTags())
    
    dmtags = getTagsFromDM3(dm)
    tags.update(dmtags)
    tags['shape'] = data.shape
    tags['original_metadata'] = si.tags
    if 'data_type' not in tags:
        tags['data_type'] = 'unknown'
    print('Found ',tags['data_type'],' in dm3 file')
    
    if tags['data_type'] == 'image':
        tags['data'] = data
        ## Find annotations
        annotations = {}
        for key in si.tags:
            if 'AnnotationGroupList' in key:
                #print(key, dict(current_channel.attrs)[key])
                split_keys= key.split('.')
                if split_keys[4] not in annotations:
                    annotations[split_keys[4]] = {}
                if split_keys[5] in ['AnnotationType','Text','Rectangle','Name', 'Label']:
                    annotations[split_keys[4]][split_keys[5]]=si.tags[key]
        tags['annotations'] = {}
        for key in annotations:
            if annotations[key]['AnnotationType']==13: 
                tags['annotations'][key] = {}
                if 'Label' in annotations[key]:
                    tags['annotations'][key]['label'] = annotations[key]['Label']
                tags['annotations'][key]['type'] = 'text'
                rect = np.array(annotations[key]['Rectangle'])* np.array([tags['axis']['0']['scale'],tags['axis']['1']['scale'],tags['axis']['0']['scale'],tags['axis']['1']['scale']])
                tags['annotations'][key]['x'] = rect[1]
                tags['annotations'][key]['y'] = rect[0]
                tags['annotations'][key]['text'] = annotations[key]['Text']

            elif annotations[key]['AnnotationType']==6:
                tags['annotations'][key] = {}
                if 'Label' in annotations[key]:
                    tags['annotations'][key]['label'] = annotations[key]['Label']
                tags['annotations'][key]['type'] = 'circle'
                rect = np.array(annotations[key]['Rectangle'])* np.array([tags['axis']['0']['scale'],tags['axis']['1']['scale'],tags['axis']['0']['scale'],tags['axis']['1']['scale']])
                
                tags['annotations'][key]['radius'] =rect[3]-rect[1]
                tags['annotations'][key]['position'] = [rect[1],rect[0]]
                
                

            elif annotations[key]['AnnotationType']==23:
                if 'Name' in annotations[key]:
                    if annotations[key]['Name'] == 'Spectrum Image':
                        tags['annotations'][key] = {}
                        if 'Label' in annotations[key]:
                            tags['annotations'][key]['label'] = annotations[key]['Label']
                        tags['annotations'][key]['type'] = 'spectrum image'
                        rect = np.array(annotations[key]['Rectangle'])* np.array([tags['axis']['0']['scale'],tags['axis']['1']['scale'],tags['axis']['0']['scale'],tags['axis']['1']['scale']])

                        tags['annotations'][key]['width'] =rect[3]-rect[1]
                        tags['annotations'][key]['height'] =rect[2]-rect[0]
                        position = tags['annotations'][key]['position'] = [rect[1],rect[0]]
                        
    elif tags['data_type'] == 'image_stack':
        tags['data'] = data[:,:,0]
        tags['cube'] = data
    elif tags['data_type'] == ['spectrum_image']:
        tags['spec'] = data[0,0,:]
        tags['SI'] = {}
        tags['SI']['data'] = data
        tags['cube'] = data
    elif tags['data_type'] == 'EELS_linescan':
        ## A linescan is a spectrum image with the second spatial dimension being 1
        ## The third dimension is the spectrum
        if tags['axis']['0']['units']=='eV':
            data = data.T
            tags['axis']['2']= tags['axis']['0'].copy()
            tags['axis']['0']= tags['axis']['1'].copy()
        else:
            tags['axis']['2']= tags['axis']['1'].copy()
            tags['axis']['1']= tags['axis']['0'].copy()

        for key in tags['axis']['0']:
                temp  = tags['axis']['2'][key]
                tags['axis']['2'][key] = tags['axis']['0'][key]
                tags['axis']['0'][key] = temp
        data = np.reshape(data,(data.shape[0],1,data.shape[1]))    
        tags['spec'] = data[0,0,:]
        tags['data'] = data
        tags['cube'] = data
        tags['SI'] = {}
        tags['SI']['data'] = data

        
    elif tags['data_type'] == 'EELS_spectrum':
        tags['spec'] = data
        tags['data'] = data
        tags['cube'] = data
    else:
        tags['data'] = data
        tags['cube'] = data
        print('OOPSY-Dasy that is not handled correctly') #Gerd!

def getTagsFromDM3 (dm):
    tags = {}
    ## Fix annoying scale of SI in Zeiss
    for key in  dm['ImageData']['Calibrations']['Dimension']:
        if dm['ImageData']['Calibrations']['Dimension'][key]['Units'] == 'µm':
            dm['ImageData']['Calibrations']['Dimension'][key]['Units'] = 'nm'
            dm['ImageData']['Calibrations']['Dimension'][key]['Scale'] *= 1000.0
    
    #Determine type of data  by 'Calibrations' tags 
    if len(dm['ImageData']['Calibrations']['Dimension']) >1:
        units1 = dm['ImageData']['Calibrations']['Dimension']['1']['Units']
    else:
        units1=''
     
    units = dm['ImageData']['Calibrations']['Dimension']['0']['Units']
    
    if 'ImageTags' in dm:
        if 'SI' in dm['ImageTags']:
            if len(dm['ImageData']['Calibrations']['Dimension']) == 3:
                tags['data_type'] = 'spectrum_image'
            else:
                if units == 'eV' or units1 == 'eV':
                    tags['data_type'] = 'EELS_linescan'
                    
                else:
                    tags['data_type'] = 'image'
                    tags['image_type'] = 'survey image'
        elif 'EELS' in dm['ImageTags']:
            tags['data_type'] = 'EELS_spectrum'
        elif len(dm['ImageData']['Calibrations']['Dimension']) == 3:
            tags['data_type'] = 'image_stack'
        else:
            tags['data_type'] = 'image'
            
        tags['microscope']= 'unknown'
    
        ## General Information from Zeiss like TEM
        if 'Microscope Info' in dm['ImageTags']:
            if 'Microscope' in dm['ImageTags']['Microscope Info']:
                tags['microscope'] =  (dm['ImageTags']['Microscope Info']['Microscope'])
                if 'Libra' in tags['microscope']:
                    tags['microscope'] = 'Libra 200'
            if 'Voltage'  in dm['ImageTags']['Microscope Info']:
                tags['acceleration_voltage']  = dm['ImageTags']['Microscope Info']['Voltage']
    
        ## General Information from Nion STEM
        if 'ImageRonchigram' in dm['ImageTags']:
            tags['image_type']= 'Ronchigram'
            tags['microscope'] = 'UltraSTEM'    
        if 'SuperScan' in  dm['ImageTags']:
            tags['microscope'] = 'UltraSTEM'
            if 'ChannelName' in dm['ImageTags']['SuperScan']:
                tags['image_type'] = dm['ImageTags']['SuperScan']['ChannelName']
                tags['detector_type'] = tags['image_type']
                tags['image_mode'] = 'STEM'
                
                    
            if 'PixelTime(us)' in dm['ImageTags']['SuperScan']:
                tags['seconds per pixel'] = dm['ImageTags']['SuperScan']['PixelTime(us)'] *1e6
            if 'BlankTime(us)' in dm['ImageTags']['SuperScan']:
                tags['seconds per flyback'] = dm['ImageTags']['SuperScan']['BlankTime(us)'] *1e6   
    
            
        if 'ImageScanned' in  dm['ImageTags']:
            if 'EHT' in dm['ImageTags']['ImageScanned']:
                tags['acceleration_voltage'] = float(dm['ImageTags']['ImageScanned']['EHT'])
                if tags['acceleration_voltage'] >100000:
                    tags['microscope'] = 'UltraSTEM 200'
                else:
                    tags['microscope'] = 'UltraSTEM 100'
            if tags['detector_type'] == 'HAADF':
                if 'PMTDF_gain' in dm['ImageTags']['ImageScanned']:
                    tags['detector_gain'] = dm['ImageTags']['ImageScanned']['PMTDF_gain']
            else:
                if 'PMTBF_gain' in dm['ImageTags']['ImageScanned']:
                    tags['detector_gain'] = dm['ImageTags']['ImageScanned']['PMTBF_gain']
            if 'StageOutX' in dm['ImageTags']['ImageScanned']:
                tags['stage'] = {}
                tags['stage']['x'] =  dm['ImageTags']['ImageScanned']['StageOutX']
                tags['stage']['y'] =  dm['ImageTags']['ImageScanned']['StageOutY']
                tags['stage']['z'] =  dm['ImageTags']['ImageScanned']['StageOutZ']
                
                tags['stage']['alpha'] =  dm['ImageTags']['ImageScanned']['StageOutA']
                tags['stage']['beta'] =  dm['ImageTags']['ImageScanned']['StageOutB']
    

    #if tags['data_type'] == 'Image':

        
    ##### Image type data in Zeiss like microscope
    tags['aberrations'] = {}
    if 'data_type' in tags:
        if tags['data_type'] == 'Image':
            if 'Microscope Info' in dm['ImageTags']:
                if units=='':
                    if 'Illumination Mode' in dm['ImageTags']['Microscope Info']:
                        if tags['ImageTags']['Microscope Info']['Illumination Mode'] == 'STEM':
                            tags['image_type']= 'Ronchigram'
                if units[:2] == '1/' or units =='mrad' :
                    tags['image_type'] = 'Diffraction'
                else:
                    tags['image_type'] = 'Image'
    
                if 'Microscope' in dm['ImageTags']['Microscope Info']:
                    tags['microscope'] =  (dm['ImageTags']['Microscope Info']['Microscope'])
                if 'Operation Mode' in dm['ImageTags']['Microscope Info']:
                    if dm['ImageTags']['Microscope Info']['Operation Mode'] =='SCANNING':
                        tags['image_mode'] = 'STEM'
                    else:
                        tags['image_mode'] = 'TEM'
                if 'Cs(mm)'in dm['ImageTags']['Microscope Info']:
                    tags['aberrations']['C30'] = float(dm['ImageTags']['Microscope Info']['Cs(mm)'])*1e6

        
    ##### Image type data in Nion microscope
    aberrations = ['C10','C12','C21','C23','C30','C32','C34','C41','C43','C45','C50','C52','C54','C56']
    assume_corrected = ['C10','C12a','C12b']
    debug = 0
    if 'ImageTags' in dm:
        if 'ImageScanned' in dm['ImageTags']:
            for key in aberrations:
                if key in dm['ImageTags']['ImageScanned']:
                    if  isinstance(dm['ImageTags']['ImageScanned'][key],dict): # if element si a dictionary
                        tags['aberrations'][key+'a'] = dm['ImageTags']['ImageScanned'][key]['a']
                        tags['aberrations'][key+'b'] = dm['ImageTags']['ImageScanned'][key]['b']
                    else:
                        tags['aberrations'][key] = dm['ImageTags']['ImageScanned'][key]
                
    #for key in tags['SuperScan']:
    #    if key == 'Rotation':
    #        print(key, tags['SuperScan'][key], ' = ' , np.rad2deg(tags['SuperScan'][key]))#/np.pi*180)
    
            if tags['acceleration_voltage'] == 200000:
                tags['aberrations']['source_size'] = 0.051
            elif tags['acceleration_voltage'] == 100000:
                tags['aberrations']['source_size'] = 0.061
            elif tags['acceleration_voltage'] == 60000:
                tags['aberrations']['source_size'] = 0.081
        
            tags['aberrations']['zeroLoss'] = [0.0143,0.0193,0.0281,0.0440,0.0768,	0.1447,	0.2785,	0.4955,	0.7442,	0.9380,	1.0000,	0.9483,	0.8596,	0.7620,	0.6539,	0.5515,0.4478,	0.3500,	0.2683,	0.1979,	0.1410,	0.1021,	0.0752,	0.0545,	0.0401,	0.0300,	0.0229,	0.0176,	0.0139]
            tags['aberrations']['zeroEnergy'] = np.linspace(-.5,.9,len(tags['aberrations']['zeroLoss']))

        tags['axis']=dm['ImageData']['Calibrations']['Dimension']
        for dimension in tags['axis']:
            for key in tags['axis'][dimension]:
                if key.lower() == 'origin':
                    tags['axis'][dimension]['offset'] = tags['axis'][dimension].pop(key)    
                else:
                    tags['axis'][dimension][key.lower()] = tags['axis'][dimension].pop(key)

            #tags['axis'][dimension]['pixels'] = tags['shape'][int(dimension)]
        if 'SuperScan' in dm['ImageTags']:  
            if 'nmPerLine' in dm['ImageTags']['SuperScan']['Calibration']:
                tags['axis']['0']['scale']  = dm['ImageTags']['SuperScan']['Calibration']['nmPerPixel']
                tags['axis']['0']['units'] = 'nm'
                tags['axis']['0']['offset'] = 0
                tags['axis']['1']['scale']  = dm['ImageTags']['SuperScan']['Calibration']['nmPerLine']
                tags['axis']['1']['units'] = 'nm'
                tags['axis']['1']['offset'] = 0
    if 'axis' in tags:    
        tags['pixel_size'] = tags['axis']['0']['scale']
        tags['FOV'] = tags['axis']['0']['scale']*dm['ImageData']['Dimensions']['0']
    
    
        for key in tags['axis']:
            if tags['axis'][key]['units'] == 'eV':
                
                tags['dispersion'] = float(tags['axis'][key]['scale'])
                tags['offset'] = -float(tags['axis'][key]['offset'])  *tags['dispersion']                
                tags['axis'][key]['offset'] = tags['offset']
    if 'spectrum_image' in dm:
        tags['spectrum_image'] = dm['spectrum_image']
        print('spectrum_image')
    if 'ImageTags' in dm:    
        if 'EELS' in dm['ImageTags']:
            eels = dm['ImageTags']['EELS']
            #print(eels)
            if 'Exposure (s)' in eels['Acquisition']:
                
                if 'Number of frames' in eels['Acquisition']:
                    tags['integration_time'] = eels['Acquisition']['Exposure (s)']*eels['Acquisition']['Number of frames']
                    tags['number_of_frames'] = eels['Acquisition']['Number of frames']
            
        #if 'Dispersion (eV/ch)' in eels:
        #    tags['dispersion'] = float(eels['Dispersion (eV/ch)'])
        #if 'Energy loss (eV)' in eels:
        #    tags['offset'] = float(eels['Energy loss (eV)'])
        #    #Gatan measures offset at channel 100, but QF at channel 1
        #    tags['offset'] = tags['offset']- 100.0*tags['dispersion']
            
            if  'Convergence semi-angle (mrad)' in eels:
                tags['convAngle']= float(eels['Convergence semi-angle (mrad)'])
            if 'Collection semi-angle (mrad)' in eels:
                tags['collAngle'] = float(eels['Collection semi-angle (mrad)'])
    
            ## Determine angles for Zeiss Libra 200 MC at UTK
            if tags['microscope'] == 'Libra 200 MC':
                
                if 'Microscope Info' in dm['ImageTags']:
                    if 'Illumination Mode' in dm['ImageTags']['Microscope Info']:
                        if dm['ImageTags']['Microscope Info']['Illumination Mode'] == 'STEM':
                            if dm['ImageTags']['Microscope Info']['Operation Mode'] == 'SCANNING':
                                if 'STEM Camera Length' in dm['ImageTags']['Microscope Info']:
                                    tags['collAngle'] = np.tan(0.65/dm['ImageTags']['Microscope Info']['STEM Camera Length'])*1000*23.73527943
                                    tags['convAngle']= 9
                                    tags['STEM camera length'] = dm['ImageTags']['Microscope Info']['STEM Camera Length']

    
    return tags

def open_qf3_file(file_name,tags):
    pkl_file = open(file_name,'rb')    
    qf= pickle.load(pkl_file)
    pkl_file.close()
    
    
    if 'QF' in qf:
        if qf['QF']['version'] < 0.982:
            #we need to update of dictionary structure
            print('updating file to new format')
            if 'DM' in qf['mem']:
                dmtemp  = qf['mem'].pop('DM', None)
                
                dm = getDictionary(dmtemp)
                outtags = getTagsFromDM3 (dm)
                
            outtags.update(qf['mem'])    
            tags.update(outtags)    

            if 'pixel_size' in tags:
                tags['pixel_size'] = tags['pixel_size']
                for key in tags['axis']:
                    if tags['axis']['0']['units'] == 'nm':
                        tags['axis']['0']['scale'] = tags['pixel_size']
                    
                tags.pop('pixel_size')
            if 'data' not in tags:
                tags['data'] = tags['spec']
            for dimension in tags['axis']:
                tags['axis'][dimension]['pixels'] = tags['data'].shape[int(dimension)]
        
            tags['original_metadata'] = dm
            tags.pop('cal', None)
            tags.pop('dm', None)
            tags.pop('EELS', None)
            tags.pop('Calibrations', None)
            tags.pop('DZM', None)

            tags.pop('Microscope', None)
            tags.pop('Nion', None)
            tags.pop('SI', None)
            tags.pop('DM', None)

            if 'SI' in qf['QF']:
                si  = qf['QF']['SI']
                
                try:
                    si.pop('specTags')
                    tags['SI'] = si
                except:
                    pass
                
            
        else:
            print('new file format')
            tags = qf
    
    else:
        tags.update(qf) ## This is a CAAC file and we need to do soemthing about that.
    #iplot_tags(tags, which = 'new')
    print(tags.keys())

    
    if tags['data_type'] == 'image':
        
    
        if 'cube' in tags:
            tags['shape'] = tags['cube'].shape
        else:
            tags['shape'] = tags['data'].shape
        
           
        tags['circs']=[]
        if 'summed' in qf:
            qf['data'] = tags['summed']
        if 'pixel_size' in tags:
            tags['pixel_size'] = tags['pixel_size']
        if 'FOV' not in tags:
            tags['FOV'] = tags['data'].shape[0]*tags['pixel_size']
        
        print ('Field of View: {0:.2f}nm'.format( tags['FOV']) )
    print('in qf3')
    print(tags['data_type'])
   
    return tags

###
#  visualization
##
# 	
	
def h5_tree(h5_file):
    """
    Just a wrapper for the usid function print_tree,
    so that usid does not have to be loaded in notebook
    """
    usid.hdf_utils.print_tree(h5_file)

def h5_plot_image(dset, ax):
    ## spatial data
    if 'image' not in dset.attrs['data_type']:
        return
    

    if ax == None:
        ax = fig.add_subplot(1,1,1)
    
    extent = dset.attrs['image_extent'] 
    
    ax.set_title('image: '+dset.attrs['title'])

    ax.imshow(dset.T)#,extent= extent)
    ax.set_xlabel(dset.dims[0].keys()[0]);

    
    annotation_done = []

    for key in dset.attrs:
        if 'annotations' in key:
            annotation_number = key[12]
            if annotation_number not in annotation_done:
                annotation_done.append(annotation_number)

                if tags['annotations_'+annotation_number+'_type'] == 'text':
                    x =tags['annotations_'+annotation_number+'_x'] 
                    y = tags['annotations_'+annotation_number+'_y']
                    text = tags['annotations_'+annotation_number+'_text'] 
                    ax.text(x,y,text,color='r')

                elif tags['annotations_'+annotation_number+'_type'] == 'circle':
                    radius = 20 * scaleX#tags['annotations'][key]['radius']
                    xy = tags['annotations_'+annotation_number+'_position']
                    circle = patches.Circle(xy, radius, color='r',fill = False)
                    ax.add_artist(circle)

                elif tags['annotations_'+annotation_number+'_type'] == 'spectrum image':
                    width = tags['annotations_'+annotation_number+'_width'] 
                    height = tags['annotations_'+annotation_number+'_height']
                    position = tags['annotations_'+annotation_number+'_position']
                    rectangle = patches.Rectangle(position, width, height, color='r',fill = False)
                    ax.add_artist(rectangle)
                    ax.text(position[0],position[1],'Spectrum Image',color='r')
    
def h5_plot(dset ,ax=None, ax2=None):
    ## Start plotting
    # tags = dict(current_channel.attrs)
    #basename = current_channel['title'][()]
    tracker = None
    if ax == None:
        if  dset.attrs['data_type'] in ['spectrum_image','image_stack']:   
            fig = plt.figure()       
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1)

            
    # plot according to what data type is in your file
    if dset.attrs['data_type'] == 'EELS_spectrum':
        if ax == None:
            ax = fig.add_subplot(1,2,1)
        ## spectrall data
        for dim in dset.dims:
            pass
        ax.plot(dset.dim['energy_scale'],dset);
        ax.set_title('spectrum: '+basename)
        ax.set_xlabel('energy loss [eV]')
        ax.set_ylim(0);
    elif dset.attrs['data_type'] == 'image':
        h5_plot_image(dset, ax)
    
        
    elif dset.attrs['data_type'] == 'spectrum_image':
        tracker = h5_spectrum_image(dset, horizontal = True)
    elif dset.attrs['data_type'] == 'image_stack':
        tracker = h5_slice_viewer(dset)
        
    return tracker

class h5_slice_viewer(object):
    def __init__(self,  dset):
    
        if dset.attrs['data_type'] != 'image_stack':
            return
        
        extent = dset.attrs['image_extent'] 
        
        self.X = dset
        if len(self.X.shape) <3:
            return
        
        if len(self.X.shape) ==3:
            rows, cols, self.slices = self.X.shape
        if len(self.X.shape) ==4:
            rows, cols, self.slices = self.X.shape
        
        self.ax = plt.axes([0.0, 0.15, .8, .8])
        self.ind = 0
        self.im = self.ax.imshow(self.X[..., self.ind].T, extent = extent)
        
        axidx = plt.axes([0.1, 0.05, 0.6, 0.03])
        self.slider = Slider(axidx, 'image', 0, self.X.shape[2]-1, valinit=self.ind, valfmt='%d')
        self.slider.on_changed(self.onSlider)
        
        self.ax.set_title('image stack: '+dset.attrs['title']+'\n use scroll wheel to navigate images')
        self.im.axes.figure.canvas.mpl_connect('scroll_event', self.onscroll)
        self.ax.set_xlabel(dset.dims[0].keys()[0]);

        self.update()

    def onSlider(self, val):
        self.ind = int(self.slider.val+0.5)
        self.slider.valtext.set_text(f'{self.ind}')
        self.update()
        
    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.ind = int(self.ind)
        self.slider.set_val(self.ind)
        #self.update()

    def update(self):
        self.im.set_data(self.X[..., int(self.ind)].T)
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw_idle()

class h5_spectrum_image(object):
    """    
    
    ### Interactive spectrum imaging plot
    
    
    """
    
    def __init__(self, dset, horizontal = True):
        
        
        if dset.attrs['data_type'] != 'spectrum_image':
            return
        print(dset.attrs['data_type'])

        box_layout = widgets.Layout(display='flex',
                    flex_flow='row',
                    align_items='stretch',
                    width='100%')

        self.figure = plt.gcf()
        self.horizontal = horizontal
        self.x = 0
        self.y = 0
        
        sizeX = dset.shape[0]
        sizeY = dset.shape[1]
        
        

        self.energy_scale = dset.dims[2][0]

        
        
        self.extent = [0,sizeX,sizeY,0]
        self.rectangle = [0,sizeX,0,sizeY]
        self.scaleX = 1.0
        self.scaleY = 1.0
        self.analysis = []
        self.plot_legend = False
        
        self.SI = False
        
        if horizontal:
            self.ax1=plt.subplot(1, 2, 1)
            self.ax2=plt.subplot(1, 2, 2)
        else:
            self.ax1=plt.subplot(2, 1, 1)
            self.ax2=plt.subplot(2, 1, 2)
            
        self.cube =  np.array(dset)
        self.image = self.cube.sum(axis=2)
        
        self.ax1.imshow(self.image.T, extent = self.extent)
        if horizontal:
            self.ax1.set_xlabel('distance [pixels]')
        else:
            self.ax1.set_ylabel('distance [pixels]')
        self.ax1.set_aspect('equal')
        
        self.rect = patches.Rectangle((0,0),1,1,linewidth=1,edgecolor='r',facecolor='red', alpha = 0.2)
        self.ax1.add_patch(self.rect)
        self.intensity_scale = 1.
        self.spectrum = self.cube[self.x, self.y, :]* self.intensity_scale
        
        
        self.ax2.plot(self.energy_scale,self.spectrum)
        self.ax2.set_title(f' spectrum {self.x},{self.y} ')
        self.ax2.set_xlabel('energy loss [eV]')
        self.ax2.set_ylabel('intensity [a.u.]')
        self.cid = self.ax2.figure.canvas.mpl_connect('button_press_event', self.onclick)
        
        
        plt.tight_layout()
    
                
                
    def onclick(self,event):
        x = int(event.xdata)
        y = int(event.ydata)
        
        if x >= self.rectangle[0] and x < self.rectangle[0]+self.rectangle[1]:
            if y >= self.rectangle[2] and y < self.rectangle[2]+self.rectangle[3]:
                self.x = int((x - self.rectangle[0])/ self.rectangle[1]*self.cube.shape[1])
                self.y = int((y - self.rectangle[2])/ self.rectangle[3]*self.cube.shape[0])
            else:
                return
        else:
            return
        
        
        self.ax2.set_xlabel(f' {x}, self {self.y}')
        self.ax2.set_xlabel(f' {x}, - {y}')
        if event.inaxes in [self.ax1]:
            self.ax2.set_xlabel(f'in axis, {x}, {y}')
            x = (self.x * self.rectangle[1]/self.cube.shape[1]+ self.rectangle[0])
            y = (self.y * self.rectangle[3]/self.cube.shape[0]+ self.rectangle[2])
            
            self.rect.set_xy([x,y]) 
            xlim = self.ax2.get_xlim()
            ylim = self.ax2.get_ylim()
            self.ax2.clear()
            self.intensity_scale = 1.
            self.spectrum = self.spectrum = self.cube[self.y, self.x, :]* self.intensity_scale
            #self.energy_scale = tags['spectra'][f'{self.x}-{self.y}']['energy_scale']
            
            self.ax2.plot(self.energy_scale,self.spectrum, label = 'experiment')
            self.ax2.set_title(f' spectrum {self.x},{self.y} ')
                
            if self.plot_legend:
                self.ax2.legend(shadow=True);
            self.ax2.set_xlim(xlim)
            self.ax2.set_ylim(ylim)
            self.ax2.set_xlabel('energy loss [eV]')
            self.ax2.set_ylabel('intensity [a.u.]')
            
            
        self.ax2.draw()
        
   
    
    def set_legend(self, setLegend):
        self.plot_legend = setLegend
    
    def get_xy(self):
        return [self.x,self.y]
    
    def get_current_spectrum(self):
        return self.cube[self.y,self.x,:]
    

###
# Crystal Structure Read and Write
###
def h5_add_crystal_structure(h5_file, crystal_tags):
    structure_group = usid.io.hdf_utils.create_indexed_group(h5_file,'Structure')
    
    structure_group['unit_cell'] = crystal_tags['unit_cell']
    structure_group['relative_positions'] = crystal_tags['base']
    structure_group['title'] = str(crystal_tags['crystal_name'])
    structure_group['_'+crystal_tags['crystal_name']] = str(crystal_tags['crystal_name'])
    structure_group['elements'] = np.array(crystal_tags['elements'],dtype='S')
    if 'zone_axis' in structure_group:
        structure_group['zone_axis'] = np.array(crystal_tags['zone_axis'], dtype=float)
    else:
        structure_group['zone_axis'] = np.array([1.,0.,0.], dtype=float)
    h5_file.flush()
    return structure_group

def h5_get_crystal_structure(structure_group):
    crystal_tags = {}
    crystal_tags['unit_cell'] = structure_group['unit_cell'][()]
    crystal_tags['base'] = structure_group['relative_positions'][()]
    crystal_tags['crystal_name'] = structure_group['title'][()]
    if '2D' in structure_group:
        crystal_tags['2D'] = structure_group['2D'][()]
    elements = structure_group['elements'][()]
    crystal_tags['elements'] = []
    for e in elements:
        crystal_tags['elements'].append( e.astype(str, copy=False))
    
    if 'zone_axis' in structure_group:
        crystal_tags['zone_axis'] = structure_group['zone_axis'] [()]
    return crystal_tags

def add_registration(current_channel, tags):
    current_dataset = current_channel['nDim_Data']
    out_tags = tags
    out_tags['analysis'] = out_tags['name']
    
    out_tags['spatial_origin_x'] = 0.
    out_tags['spatial_origin_y'] = 0.
    out_tags['spatial_scale_x'] = current_dataset.dims[0][0][1]-current_dataset.dims[0][0][0]
    out_tags['spatial_scale_y'] = current_dataset.dims[1][0][1]-current_dataset.dims[1][0][0]
    out_tags['spatial_size_x'] = out_tags['data'].shape[0]
    out_tags['spatial_size_y'] = out_tags['data'].shape[1]
    out_tags['spatial_units'] = 'nm'

    ## Log data
    out_tags['title'] = out_tags['name']
    stack_channel = log_results(current_channel, out_tags)
    return stack_channel

def add_spectrum_image(current_channel, tags):
    current_dataset = current_channel['nDim_Data']
    out_tags = tags
    new_energy_scale = out_tags['new_energy_scale']
    out_tags['analysis'] = out_tags['name']
    
    out_tags['spatial_origin_x'] = 0.
    out_tags['spatial_origin_y'] = 0.
    out_tags['spatial_scale_x'] = current_dataset.dims[0][0][1]-current_dataset.dims[0][0][0]
    out_tags['spatial_scale_y'] = current_dataset.dims[1][0][1]-current_dataset.dims[1][0][0]
    out_tags['spatial_size_x'] = out_tags['data'].shape[0]
    out_tags['spatial_size_y'] = out_tags['data'].shape[1]
    out_tags['spatial_units'] = 'nm'

    out_tags['spectral_size_x'] = tags['data'].shape[2]
    current_dataset.dims[0][0][1]-current_dataset.dims[0][0][0]
    out_tags['spectral_offset_x'] = new_energy_scale[0]
    out_tags['spectral_scale_x'] = new_energy_scale[1]-new_energy_scale[0]
    ## Log data
    out_tags['title'] = out_tags['name']
    si_channel = log_results(current_channel, out_tags)
    return si_channel

def complete_registration(current_dataset, current_channel):
    ## relative drift
    RigReg ,drift = Rigid_Registration(current_dataset)
    RigReg_crop, crop  = crop_image_stack(RigReg, drift)

    # Log Rigid Registration
    out_tags = {}
    out_tags['Rigid_registration_drift']=drift
    out_tags['Rigid_registration_crop'] = crop
    out_tags['data'] = RigReg_crop
    out_tags['data_type'] = 'image_stack'
    out_tags['name'] = 'Rigid Registration'
    out_tags['notebook']= __notebook__ 
    out_tags['notebook_version']= __notebook_version__

    stack_group = ft.add_registration(current_channel, out_tags)
    
    print('Non-Rigid_Registration')
    current_dataset = stack_group['nDim_Data']
    non_rigid_registered = DemonReg(current_dataset)

    ## Log Non-Rigid Registration
    out_tags={}
    out_tags['data'] = non_rigid_registered
    out_tags['data_type'] = 'image_stack'
    out_tags['name'] = 'Non-Rigid Registration'
    out_tags['notebook']= __notebook__ 
    out_tags['notebook_version']= __notebook_version__

    stack_group = ft.add_registration(current_channel, out_tags)
    
    return stack_group



def h5_add_diffraction(current_channel, crystal_tags):
    out_tags = {}
    out_tags['analysis']='diffraction'
    for key in crystal_tags:
        
        if not isinstance(crystal_tags[key],dict):
            if key == 'elements':
                out_tags['elements'] = np.array(crystal_tags['elements'],dtype='S')
            elif key in ['crystal_name','symmetry','reference','link']:
                out_tags[key] = str(crystal_tags[key])
            elif key in ['label']:
                pass # don't know how to write that format
            else:
                if key == 'label':
                    pass# don't know how to write that format
                out_tags[key] = np.array(crystal_tags[key])
        else:
            if key == 'allowed':
                for key2 in crystal_tags['allowed']:
                    if key2 != 'label':
                        out_tags[key2] = np.array(crystal_tags['allowed'][key2])
    log_group = h5_add_Log(current_channel, crystal_tags['crystal_name']+' - '+str(crystal_tags['zone_hkl']))
    h5_add_Data2Log(log_group, out_tags)
    return log_group

