"""buffers.py

Classes to hold references to a large ADCIRC ASCII record file. Due to the
large size of these files, data is read in one step at a time.
"""

import numpy

class TimeSeriesBuffer:
    """Wrapper for an ADCIRC time series file buffer.

    This struct is a convenience utility for reading large the large ASCII
    outputs from ADCIRC. Instead of loading the whole array into memory at once,
    the data is accessed one time step at a time.
    """

    def __init__(self, path: str):
        """Constructor.

        Arguments:
            path (str): A path to the data file.
        """
        self._path = path
        self._f = None # Wait to open the file
        self._NDSETS = None # Number of records
        self._NP = None # Number of nodes
        self._DTDP_NSPOOL = None # Time increment between writes
        self._NSPOOL = None # Iteration increment between writes
        self._IRTYPE = None # Record type
        self._num_read = 0 # Counter for records

    def __enter__(self):
        """Called when the oject is invoked in a with statement.

        For example, with TimeSeriesBuffer('fort.63') as b, the result of this
        function is passed to b.
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when leaving a context manager. This ensures that the file
        object is closed.
        """
        self.close()

    def __iter__(self):
        """Called in a for-each loop. Returns an iterable object."""
        return self

    def __next__(self):
        """Get the next iteration, i.e. record in the file."""
        # If every record has been read, exit
        if self._num_read == self._NDSETS:
            raise StopIteration
        return self.read_step()

    def open(self):
        """Open the file."""
        self._f = open(self._path, 'r')
        self._f.readline() # Skip descriptive header

        # Read data set metadata
        (self._NDSETS, self._NP, self._DTDP_NSPOOL, self._NSPOOL,
            self._IRTYPE) = numpy.loadtext(self._f, dtype=float, max_rows=1,
            usecols=(0, 1, 2, 3, 4))
        self._num_read = 0

    def read_step(self, node_mask: numpy.ndarray = None
        ) -> tuple[float, float, numpy.ndarray]:
        """Read a single time step.

        Arguments:
            node_mask (numpy.ndarray, optional): A boolean array to filter to
                a subset of nodes.

        Returns:
            tuple[float, float, numpy.ndarray]
            time (float): The time associated with this record.
            it (float): The iteration numer associated with this record. Note
                that this is derived from the solution loop iteration, so it is
                not necessarily 0, 1, 2...
            data (numpy.ndarray): The nodal data assocaited with the record.
            """
        time, it = numpy.loadtext(self._f, dtype=float, max_rows=1,
            usecols=(0, 1))
        data = numpy.loadtext(self._f, dtype=float, max_rows=self._NP)
        data = data[1:] # Filter out k, this is not the same as JN
        self._num_read += 1
        if node_mask is not None:
            data = data[node_mask, :]
        return time, it, data

    def close(self):
        """Close the file."""
        self._f.close()
