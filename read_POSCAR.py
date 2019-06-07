from ase.io import read, write
import pyTEMlib.file_tools as ft

def Read_POSCAR(): # open file dialog to select poscar file
    file_name = ft.openfile_dialog('POSCAR (POSCAR*.txt);;All files (*)')
    #use ase package to read file
    crystal = read(file_name,format='vasp', parallel=False)

    ## make dictionary and plot structure (not essential for further notebook)
    tags = {}
    tags['unit_cell'] = crystal.cell*1e-1
    tags['elements'] = crystal.get_chemical_symbols()
    tags['base'] = crystal.get_scaled_positions()

    tags['max_bond_length'] = 0.23
    tags['name'] = 'crystal'

    return tags

