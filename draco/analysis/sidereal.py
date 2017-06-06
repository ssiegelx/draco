"""Take timestream data and regridding it into sidereal days which can be
stacked.

Tasks
=====

.. autosummary::
    :toctree:

    SiderealGrouper
    SiderealRegridder
    SiderealStacker

Usage
=====

Generally you would want to use these tasks together. Sending time stream data
into  :class:`SiderealGrouper`, then feeding that into
:class:`SiderealRegridder` to grid onto each sidereal day, and then into
:class:`SiderealStacker` if you want to combine the different days.
"""


import numpy as np
import os
import shutil

from caput import config, mpiutil, mpiarray, tod

from ..core import task, containers
from ..util import regrid, tools


class SiderealGrouper(task.SingleTask):
    """Group individual timestreams together into whole Sidereal days.

    Attributes
    ----------
    min_day_length : float
        Only process partial sidereal days whose length is greater
        than this fraction of the whole sidereal day.
    """

    min_day_length = config.Property(proptype=float, default=0.1)

    def __init__(self):
        self._timestream_list = []
        self._current_lsd = None

    def setup(self, observer):
        """Set the local observers position.

        Parameters
        ----------
        observer : :class:`~caput.time.Observer`
            An Observer object holding the geographic location of the telescope.
            Note that :class:`~drift.core.TransitTelescope` instances are also
            Observers.
        """
        self.observer = observer

    def process(self, tstream):
        """Load in each sidereal day.

        Parameters
        ----------
        tstream : containers.TimeStream
            Timestream to group together.

        Returns
        -------
        ts : containers.TimeStream or None
            Returns the timestream of each sidereal day when we have received
            the last file, otherwise returns :obj:`None`.
        """

        # Get the start and end LSDs of the file
        lsd_start = int(self.observer.unix_to_lsd(tstream.time[0]))
        lsd_end = int(self.observer.unix_to_lsd(tstream.time[-1]))

        # If current_lsd is None then this is the first time we've run
        if self._current_lsd is None:
            self._current_lsd = lsd_start

        # If this file started during the current lsd add it onto the list
        if self._current_lsd == lsd_start:
            self._timestream_list.append(tstream)

        if tstream.vis.comm.rank == 0:
            print "Adding file into group for LSD:%i" % lsd_start

        # If this file ends during a later LSD then we need to process the
        # current list and restart the system
        if self._current_lsd < lsd_end:

            if tstream.vis.comm.rank == 0:
                print "Concatenating files for LSD:%i" % lsd_start

            # Combine timestreams into a single container for the whole day this
            # could get returned as None if there wasn't enough data
            tstream_all = self._process_current_lsd()

            # Reset list and current LSD for the new file
            self._timestream_list = [tstream]
            self._current_lsd = lsd_end

            return tstream_all
        else:
            return None

    def process_finish(self):
        """Return the final sidereal day.

        Returns
        -------
        ts : containers.TimeStream or None
            Returns the timestream of the final sidereal day if it's long
            enough, otherwise returns :obj:`None`.
        """

        # If we are here there is no more data coming, we just need to process any remaining data
        tstream_all = self._process_current_lsd()

        return tstream_all

    def _process_current_lsd(self):
        # Combine the current set of files into a timestream

        lsd = self._current_lsd

        # Calculate the length of data in this current LSD
        day_length = np.sum([max(min(lsd + 1, self.observer.unix_to_lsd(tl.time[-1])), lsd) -
                             max(min(lsd + 1, self.observer.unix_to_lsd(tl.time[0])), lsd)
                             for tl in self._timestream_list])

        if mpiutil.rank0:
            print "Day length %0.2f" % day_length

        # If the amount of data for this day is too small, then just skip
        if day_length < self.min_day_length:
            return None

        if self._timestream_list[0].vis.comm.rank == 0:
            print "Constructing LSD:%i [%i files]" % (lsd, len(self._timestream_list))

        # Construct the combined timestream
        ts = tod.concatenate(self._timestream_list)

        # Add attributes for the LSD and a tag for labelling saved files
        ts.attrs['tag'] = ('lsd_%i' % lsd)
        ts.attrs['lsd'] = lsd

        return ts


class SiderealRegridder(task.SingleTask):
    """Take a sidereal days worth of data, and put onto a regular grid.

    Uses a maximum-likelihood inverse of a Lanczos interpolation to do the
    regridding. This gives a reasonably local regridding, that is pretty well
    behaved in m-space.

    Attributes
    ----------
    samples : int
        Number of samples across the sidereal day.
    lanczos_width : int
        Width of the Lanczos interpolation kernel.
    weight_scale_factor : float
        Multiply weights by this factor to prevent overflow.
    signal_inv_var : float
        Inverse variance of signal for Weiner filter.
    """

    samples = config.Property(proptype=int, default=1024)
    lanczos_width = config.Property(proptype=int, default=5)
    weight_scale_factor = config.Property(proptype=float, default=1e-7)
    signal_inv_var = config.Property(proptype=float, default=5e-3)

    def setup(self, observer):
        """Set the local observers position.

        Parameters
        ----------
        observer : :class:`~caput.time.Observer`
            An Observer object holding the geographic location of the telescope.
            Note that :class:`~drift.core.TransitTelescope` instances are also
            Observers.
        """
        self.observer = observer

    def process(self, data):
        """Regrid the sidereal day.

        Parameters
        ----------
        data : containers.TimeStream
            Timestream data for the day (must have a `LSD` attribute).

        Returns
        -------
        sdata : containers.SiderealStream
            The regularly gridded sidereal timestream.
        """

        if mpiutil.rank0:
            print "Regridding LSD:%i" % data.attrs['lsd']

        # Redistribute if needed too
        data.redistribute('freq')

        # Convert data timestamps into LSDs
        timestamp_lsd = self.observer.unix_to_lsd(data.time)

        # Fetch which LSD this is
        lsd = data.attrs['lsd']

        # Create a regular grid in LSD, padded at either end to supress interpolation issues
        pad = 5 * self.lanczos_width
        lsd_grid = lsd + np.arange(-pad, self.samples + pad, dtype=np.float64) / self.samples

        # Construct regridding matrix
        lzf = regrid.lanczos_forward_matrix(lsd_grid, timestamp_lsd, self.lanczos_width).T.copy()

        # Mask data
        vis_data = data.vis[:].view(np.ndarray)
        imask = data.weight[:].view(np.ndarray)

        # Reshape data
        vr = vis_data.reshape(-1, vis_data.shape[-1])
        nr = imask.reshape(-1, vis_data.shape[-1])

        # Ensure that the median inverse noise variance is on the order 1.0
        nr *= self.weight_scale_factor

        # Mean subtract visibilities
        mu = (np.sum(nr * vr, axis=-1) * tools.invert_no_zero(np.sum(nr, axis=-1)))[:, np.newaxis]
        vr -= mu

        # Construct a signal 'covariance'
        Si = np.ones_like(lsd_grid) * self.signal_inv_var

        # Calculate the interpolated data and a noise weight at the points in the padded grid
        sts, ni = regrid.band_wiener(lzf, nr, Si, vr, 2 * self.lanczos_width - 1)

        # Throw away the padded ends
        sts = sts[:, pad:-pad].copy()
        ni = ni[:, pad:-pad].copy()

        # Add mean back to visibilities and rescale noise
        sts += mu * (np.abs(sts) > 0)
        ni /= self.weight_scale_factor

        # Reshape to the correct shape
        sts = sts.reshape(vis_data.shape[:-1] + (self.samples,))
        ni = ni.reshape(vis_data.shape[:-1] + (self.samples,))

        # Wrap to produce MPIArray
        sts = mpiarray.MPIArray.wrap(sts, axis=data.vis.distributed_axis)
        ni = mpiarray.MPIArray.wrap(ni, axis=data.vis.distributed_axis)

        # FYI this whole process creates an extra copy of the sidereal stack.
        # This could probably be optimised out with a little work.
        sdata = containers.SiderealStream(axes_from=data, ra=self.samples)
        sdata.redistribute('freq')
        sdata.vis[:] = sts
        sdata.weight[:] = ni
        sdata.attrs['lsd'] = lsd
        sdata.attrs['tag'] = 'lsd_%i' % lsd

        return sdata


class SiderealStacker(task.SingleTask):
    """Take in a set of sidereal days, and stack them up.

    This will apply relative calibration.

    Attributes
    ----------
    checkpoint : bool
        Save the stack to disk after adding each sidereal day.

    restart_from : str
        Restart the stack from this file.
    """

    checkpoint = config.Property(proptype=bool, default=False)
    restart_from = config.Property(proptype=str, default=None)

    stack = None

    def setup(self):
        """ If requested, restart the stack from file saved to disk.
        """

        if (self.restart_from is not None) and os.path.isfile(self.restart_from):

            if mpiutil.rank0:
                print "Restarting stack from: %s" % self.restart_from

            # Load sidereal stack from disk
            self.stack = containers.SiderealStream.from_file(self.restart_from, distributed=True)

            # Redistribute over the frequency axis
            self.stack.redistribute('freq')


    def process(self, sdata):
        """Stack up sidereal days.

        Parameters
        ----------
        sdata : containers.SiderealStream
            Individual sidereal day to stack up.
        """

        sdata.redistribute('freq')

        # Get the LSD label out of the data (resort to using a CSD if it's
        # present). If there's no label just use a place holder and stack
        # anyway.
        if 'lsd' in sdata.attrs:
            input_lsd = sdata.attrs['lsd']
        elif 'csd' in sdata.attrs:
            input_lsd = sdata.attrs['csd']
        else:
            input_lsd = -1

        input_lsd = _ensure_list(input_lsd, array=True)


        if self.stack is None:

            if mpiutil.rank0:
                print "Starting stack with LSD:  " + np.array2string(input_lsd, separator=', ')

            self.stack = containers.empty_like(sdata)
            self.stack.redistribute('freq')

            self.stack.vis[:] = sdata.vis[:] * sdata.weight[:]
            self.stack.weight[:] = sdata.weight[:]

            self.stack.attrs['tag'] = 'stack'
            self.stack.attrs['lsd'] = input_lsd

        else:

            if any([ ll in self.stack.attrs['lsd'] for ll in input_lsd ]):
                if mpiutil.rank0:
                    print "Skipping stack of LSD:  " + np.array2string(input_lsd, separator=', ')
                return

            if mpiutil.rank0:
                print "Adding to stack LSD:  " + np.array2string(input_lsd, separator=', ')

            # note: Eventually we should fix up gains

            # Combine stacks with inverse `noise' weighting
            self.stack.vis[:] += (sdata.vis[:] * sdata.weight[:])
            self.stack.weight[:] += sdata.weight[:]

            self.stack.attrs['lsd'] = np.append(self.stack.attrs['lsd'],
                                                input_lsd)


        # Checkpoint the stack
        self._save_checkpoint()


    def process_finish(self):
        """Construct and emit sidereal stack.

        Returns
        -------
        stack : containers.SiderealStream
            Stack of sidereal days.
        """

        self.stack.vis[:] *= tools.invert_no_zero(self.stack.weight[:].view(np.ndarray))

        return self.stack


    def _save_checkpoint(self):

        if self.checkpoint:

            # Create a tag for the output file name
            tag = self.stack.attrs['tag'] if 'tag' in self.stack.attrs else 'stack'

            # Construct the filename
            outfile = self.output_root + str(tag) + '.h5'

            # Expand any variables in the path
            outfile = os.path.expanduser(outfile)
            outfile = os.path.expandvars(outfile)

            # If the output file already exists, then we would like to
            # rename it temporarily while writing the contents of the stack.
            # We do not want to lose our progress if the job ends while writing.
            if mpiutil.rank0:

                delete_temp = False

                if os.path.isfile(outfile):

                    tempfile = list(os.path.splitext(outfile))
                    tempfile.insert(-1, '_temp')
                    tempfile = ''.join(tempfile)

                    shutil.move(outfile, tempfile)

                    delete_temp = True

            mpiutil.barrier()

            # Save checkpoint
            self.write_output(outfile, self.stack)

            # Finished writing stack to disk!
            # If necessary, delete temporary file.
            if mpiutil.rank0 and delete_temp:
                os.remove(tempfile)

            mpiutil.barrier()



def _ensure_list(x, array=False):

    if hasattr(x, '__iter__'):
        y = [xx for xx in x]
    else:
        y = [x]

    if array:
        y = np.array(y)

    return y