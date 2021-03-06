"""Tasks for reading and writing data.

Tasks
=====

.. autosummary::
    :toctree:

    LoadFiles
    LoadMaps
    LoadFilesFromParams
    Save
    Print
    LoadBeamTransfer

File Groups
===========

Several tasks accept groups of files as arguments. These are specified in the YAML file as a dictionary like below.

.. code-block:: yaml

    list_of_file_groups:
        -   tag: first_group  # An optional tag naming the group
            files:
                -   'file1.h5'
                -   'file[3-4].h5'  # Globs are processed
                -   'file7.h5'

        -   files:  # No tag specified, implicitly gets the tag 'group_2'
                -   'another_file1.h5'
                -   'another_file2.h5'


    single_group:
        files: ['file1.h5', 'file2.h5']
"""

import os

import numpy as np

from caput import pipeline
from caput import config

from . import task


def _list_of_filelists(files):
    # Take in a list of lists/glob patterns of filenames
    import glob

    f2 = []

    for filelist in files:

        if isinstance(filelist, str):
            filelist = glob.glob(filelist)
        elif isinstance(filelist, list):
            pass
        else:
            raise Exception('Must be list or glob pattern.')
        f2.append(filelist)

    return f2


def _list_or_glob(files):
    # Take in a list of lists/glob patterns of filenames
    import glob

    if isinstance(files, str):
        files = sorted(glob.glob(files))
        print files
    elif isinstance(files, list):
        pass
    else:
        raise RuntimeError('Must be list or glob pattern.')

    return files


def _list_of_filegroups(groups):
    # Process a file group/groups
    import glob

    # Convert to list if the group was not included in a list
    if not isinstance(groups, list):
        groups = [groups]

    # Iterate over groups, set the tag if needed, and process the file list
    # through glob
    for gi, group in enumerate(groups):

        files = group['files']

        if 'tag' not in group:
            group['tag'] = 'group_%i' % gi

        flist = []

        for fname in files:
            flist += glob.glob(fname)

        group['files'] = flist

    return groups


class LoadMaps(pipeline.TaskBase):
    """Load a series of maps from files given in the tasks parameters.

    Maps are given as one, or a list of `File Groups` (see
    :mod:`draco.core.io`). Maps within the same group are added together
    before being passed on.

    Attributes
    ----------
    maps : list or dict
        A dictionary specifying a file group, or a list of them.
    """

    maps = config.Property(proptype=_list_of_filegroups)

    def next(self):
        """Load the groups of maps from disk and pass them on.

        Returns
        -------
        map : :class:`containers.Map`
        """

        from . import containers

        # Exit this task if we have eaten all the file groups
        if len(self.maps) == 0:
            raise pipeline.PipelineStopIteration

        group = self.maps.pop(0)

        map_stack = None

        # Iterate over all the files in the group, load them into a Map
        # container and add them all together
        for mfile in group['files']:

            current_map = containers.Map.from_file(mfile, distributed=True)
            current_map.redistribute('freq')

            # Start the stack if needed
            if map_stack is None:
                map_stack = current_map

            # Otherwise, check that the new map has consistent frequencies,
            # nside and pol and stack up.
            else:

                if (current_map.freq != map_stack.freq).all():
                    raise RuntimeError('Maps do not have consistent frequencies.')

                if (current_map.index_map['pol'] != map_stack.index_map['pol']).all():
                    raise RuntimeError('Maps do not have the same polarisations.')

                if (current_map.index_map['pixel'] != map_stack.index_map['pixel']).all():
                    raise RuntimeError('Maps do not have the same pixelisation.')

                map_stack.map[:] += current_map.map[:]

        # Assign a tag to the stack of maps
        map_stack.attrs['tag'] = group['tag']

        return map_stack


class LoadFilesFromParams(pipeline.TaskBase):
    """Load data from files given in the tasks parameters.

    Attributes
    ----------
    files : glob pattern, or list
        Can either be a glob pattern, or lists of actual files.
    """

    files = config.Property(proptype=_list_or_glob)
    tag_search = config.Property(proptype=str, default=None)

    def next(self):
        """Load the given files in turn and pass on.

        Returns
        -------
        cont : subclass of `memh5.BasicCont`
        """

        from caput import memh5

        if len(self.files) == 0:
            raise pipeline.PipelineStopIteration

        # Fetch and remove the first item in the list
        file_ = self.files.pop(0)

        print "Loading file %s" % file_

        cont = memh5.BasicCont.from_file(file_, distributed=True)

        # Determine an appropriate tag for the container.
        # First check for tag in attributes.
        tag = cont.attrs.get('tag', None)

        # If tag_search keyword has been input, then obtain tag
        # from keyword search of the file's directories.
        if self.tag_search is not None:
            search_res = [tdir for tdir in os.path.normpath(file_).split(os.sep)
                               if tdir.startswith(self.tag_search)]

            if search_res:
                tag = '_'.join(search_res)

        # Otherwise, use the first part of the filename as the tag.
        if tag is None:
            tag = os.path.splitext(os.path.basename(file_))[0]

        cont.attrs['tag'] = tag

        return cont


# Define alias for old code
LoadBasicCont = LoadFilesFromParams


class LoadFiles(LoadFilesFromParams):
    """Load data from files passed into the setup routine.

    File must be a serialised subclass of :class:`memh5.BasicCont`.
    """

    files = None

    def setup(self, files):
        """Set the list of files to load.

        Parameters
        ----------
        files : list
        """
        if not isinstance(files, (list, tuple)):
            raise RuntimeError('Argument must be list of files.')

        self.files = files


class Save(pipeline.TaskBase):
    """Save out the input, and pass it on.

    Assumes that the input has a `to_hdf5` method. Appends a *tag* if there is
    a `tag` entry in the attributes, otherwise just uses a count.

    Attributes
    ----------
    root : str
        Root of the file name to output to.
    """

    root = config.Property(proptype=str)

    count = 0

    def next(self, data):
        """Write out the data file.

        Assumes it has an MPIDataset interface.

        Parameters
        ----------
        data : mpidataset.MPIDataset
            Data to write out.
        """

        if 'tag' not in data.attrs:
            tag = self.count
            self.count += 1
        else:
            tag = data.attrs['tag']

        fname = '%s_%s.h5' % (self.root, str(tag))

        data.to_hdf5(fname)

        return data


class Print(pipeline.TaskBase):
    """Stupid module which just prints whatever it gets. Good for debugging.
    """

    def next(self, input_):

        print input_

        return input_


class LoadBeamTransfer(pipeline.TaskBase):
    """Loads a beam transfer manager from disk.

    Attributes
    ----------
    product_directory : str
        Path to the saved Beam Transfer products.
    """

    product_directory = config.Property(proptype=str)

    def setup(self):
        """Load the beam transfer matrices.

        Returns
        -------
        tel : TransitTelescope
            Object describing the telescope.
        bt : BeamTransfer
            BeamTransfer manager.
        feed_info : list, optional
            Optional list providing additional information about each feed.
        """

        import os

        from drift.core import beamtransfer

        if not os.path.exists(self.product_directory):
            raise RuntimeError('BeamTransfers do not exist.')

        bt = beamtransfer.BeamTransfer(self.product_directory)

        tel = bt.telescope

        try:
            return tel, bt, tel.feeds
        except AttributeError:
            return tel, bt
