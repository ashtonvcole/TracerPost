"""mesh.py

Tools for reading ADCIRC meshes and translating them to FEniCSx.
"""

import numpy

class ADCIRCMesh:
    """Wrapper for an ADCIRC mesh object."""

    def __init__(self, AGRID: str, NE: int, NP: int, JN: numpy.ndarray,
        X: numpy.ndarray, Y: numpy.ndarray, DP: numpy.ndarray,
        JE: numpy.ndarray, NM: numpy.ndarray, NOPE: int, NETA: int,
        NVDLL: list[int], IBTYPEE: list[int], NBDV: list[numpy.ndarray],
        NBOU: int, NVEL: int, NVELL: list[int], IBTYPE: list[int],
        NBVV: list[numpy.ndarray], BARLANHT: list[numpy.ndarray],
        BARLANCFSP: list[numpy.ndarray], IBCONN: list[numpy.ndarray],
        BARINHT: list[numpy.ndarray], BARINCFSB: list[numpy.ndarray],
        BARINCFSP: list[numpy.ndarray], PIPEHT: list[numpy.ndarray],
        PIPECOEF: list[numpy.ndarray], PIPEDIAM: list[numpy.ndarray]):
        """Constructor.

        Arguments:
            AGRID (str): The grid name.
            NE (int): The number of elements in the mesh.
            NP (int): The number of nodes in the mesh.
            JN (numpy.ndarray): The ADCIRC index of each node. Shape is (NP,).
            X (numpy.ndarray): The first coordinate of each node. Shape is
                (NP,).
            Y (numpy.ndarray): The second coordinate of each node. Shape is
                (NP,),
            DP (numpy.ndarray): The bathymetry of each node. Shape is (NP,).
            JE (numpy.ndarray): The ADCIRC index of each element. Shape is
                (NE,).
            NM (numpy.ndarray): The node-element connectivity of the mesh, using
                the ADCIRC indices of nodes composing each element. Shape is
                (NE, 3).
            NOPE (int): The number of elevation boundary segments.
            NETA (int): The number of elevation boundary nodes.
            NVDLL (list[int]): The number of nodes in each elevation boundary
                segment. Length is NOPE.
            IBTYPEE (list[int]): The type for each elevation boundary segment.
                Length is NOPE.
            NBDV (list[numpy.ndarray]): The ADCIRC node indices for each
                elevation boundary segment. Length is NOPE. Each array has shape
                (NVDLL[k],).
            NBOU (int): The number of normal flow segments.
            NVEL (int): The number of normal flow nodes.
            NVELL (list[int]): The number of nodes in each normal flow boundary
                segment. Length is NBOU.
            IBTYPE (list[int]): The type of each normal flow boundary segment.
                Length is NBOU.
            NBVV (list[numpy.ndarray]): The ADCIRC node indices for each
                normal flow boundary segment. Length is NBOU. Each array has
                shape (NVELL[k],).
            BARLANHT (list[numpy.ndarray]): Barrier heights along each external
                normal flow boundary segment for each node. Length is NBOU. Each
                array has shape (NVELL[k],).
            BARLANCFSP (list[numpy.ndarray]): Coefficient for free surface
                supercritical flow along each external normal flow boundary
                segment for each node. Length is NBOU. Each array has shape
                (NVELL[k],).
            IBCONN (list[numpy.ndarray]): ADCIRC indices of the back face nodes
                paired with front face nodes, along each normal flow boundary
                segment for each node. Length is NBOU. Each array has shape
                (NVELL[k],).
            BARINHT (list[numpy.ndarray]): Barrier heights along each internal
                normal flow boundary segment for each node. Length is NBOU. Each
                array has shape (NVELL[k],).
            BARINCFSB (list[numpy.ndarray]): Coefficient for free surface
                subcritical flow along each internal normal flow boundary
                segment for each node. Length is NBOU. Each array has shape
                (NVELL[k],).
            BARINCFSP (list[numpy.ndarray]): Coefficient for free surface
                supercritical flow along each internal normal flow boundary
                segment for each node. Length is NBOU. Each array has shape
                (NVELL[k],).
            PIPEHT (list[numpy.ndarray]): Cross barrier pipe height along each
                internal normal flow boundary for each node. Length is NBOU.
                Each array has shape (NVELL[k],).
            PIPECOEF (list[numpy.ndarray]): Cross barrier pipe bulk friction
                factor along each internal normal flow boundary for each node.
                Length is NBOU. Each array has shape (NVELL[k],).
            PIPEDIAM (list[numpy.ndarray]): Cross barrier pipe diameter along
                each internal normal flow boundary for each node. Length is
                NBOU. Each array has shape (NVELL[k],).
        """
        self.AGRID = AGRID
        self.NE = NE
        self.NP = NP
        self.JN = JN
        self.X = X
        self.Y = Y
        self.DP = DP
        self.JE = JE
        self.NM = NM
        self.NOPE = NOPE
        self.NETA = NETA
        self.NVDLL = NVDLL
        self.IBTYPEE = IBTYPEE
        self.NBDV = NBDV
        self.NBOU = NBOU
        self.NVEL = NVEL
        self.NVELL = NVELL
        self.IBTYPE = IBTYPE
        self.NBVV = NBVV
        self.BARLANHT = BARLANHT
        self.BARLANCFSP = BARLANCFSP
        self.IBCONN = IBCONN
        self.BARINHT = BARINHT
        self.BARINCFSB = BARINCFSB
        self.BARINCFSP = BARINCFSP
        self.PIPEHT = PIPEHT
        self.PIPECOEF = PIPECOEF
        self.PIPEDIAM = PIPEDIAM

    @property
    def AGRID(self) -> str:
        """str: The grid name."""
        return self._AGRID

    @AGRID.setter
    def AGRID(self, value: str):
        self._AGRID = value

    @property
    def NE(self) -> int:
        """int: The number of elements in the mesh."""
        return self._NE

    @NE.setter
    def NE(self, value: int):
        self._NE = value

    @property
    def NP(self) -> int:
        """int: The number of nodes in the mesh."""
        return self._NP

    @NP.setter
    def NP(self, value: int):
        self._NP = value

    @property
    def JN(self):
        """numpy.ndarray: The ADCIRC index of each node. Shape is (NP,)."""
        return self._JN

    @JN.setter
    def JN(self, value: numpy.ndarray):
        self._JN = value

    @property
    def X(self) -> numpy.ndarray:
        """numpy.ndarray: The first coordinate of each node. Shape is (NP,)."""
        return self._X

    @X.setter
    def X(self, value: numpy.ndarray):
        self._X = value

    @property
    def Y(self) -> numpy.ndarray:
        """numpy.ndarray: The second coordinate of each node. Shape is (NP,)."""
        return self._Y

    @Y.setter
    def Y(self, value: numpy.ndarray):
        self._Y = value

    @property
    def DP(self) -> numpy.ndarray:
        """numpy.ndarray: The bathymetry of each node. Shape is (NP,)."""
        return self._DP

    @DP.setter
    def DP(self, value: numpy.ndarray):
        self._DP = value

    @property
    def JE(self) -> numpy.ndarray:
        """numpy.ndarray: The ADCIRC index of each element. Shape is (NP,)."""
        return self._JE

    @JE.setter
    def JE(self, value: numpy.ndarray):
        self._JE = value

    @property
    def NM(self) -> numpy.ndarray:
        """numpy.ndarray: The node-element connectivity of the mesh, using the
            ADCIRC indices of nodes composing each element. Shape is (NE, 3).
        """
        return self._NM

    @NM.setter
    def NM(self, value: numpy.ndarray):
        self._NM = value

    @property
    def NOPE(self) -> int:
        """int: The number of elevation boundary segments."""
        return self._NOPE

    @NOPE.setter
    def NOPE(self, value: int):
        self._NOPE = value

    @property
    def NETA(self) -> int:
        """int: The number of elevation boundary nodes."""
        return self._NETA

    @NETA.setter
    def NETA(self, value: int):
        self._NETA = value

    @property
    def NVDLL(self) -> list[int]:
        """list[int]: The number of nodes in each elevation boundary segment.
        Length is NOPE.
        """
        return self._NVDLL

    @NVDLL.setter
    def NVDLL(self, value: list[int]):
        self._NVDLL = value

    @property
    def IBTYPEE(self) -> list[int]:
        """list[int]: The type for each elevation boundary segment. Length is
        NOPE.
        """
        return self._IBTYPEE

    @IBTYPEE.setter
    def IBTYPEE(self, value: list[int]):
        self._IBTYPEE = value

    @property
    def NBDV(self) -> list[numpy.ndarray]:
        """list[numpy.ndarray]: The ADCIRC node indices for each elevation
        boundary segment. Length is NOPE. Each array has shape (NVDLL[k],).
        """
        return self._NBDV

    @NBDV.setter
    def NBDV(self, value: list[numpy.ndarray]):
        self._NBDV = value

    @property
    def NBOU(self) -> int:
        """int: The number of normal flow segments."""
        return self._NBOU

    @NBOU.setter
    def NBOU(self, value: int):
        self._NBOU = value

    @property
    def NVEL(self) -> int:
        """int: The number of normal flow nodes."""
        return self._NVEL

    @NVEL.setter
    def NVEL(self, value: int):
        self._NVEL = value

    @property
    def NVELL(self) -> list[int]:
        """list[int]: The number of nodes in each normal flow segment. Length is
        NBOU.
        """
        return self._NVELL

    @NVELL.setter
    def NVELL(self, value: list[int]):
        self._NVELL = value

    @property
    def IBTYPE(self) -> list[int]:
        """list[int]: The type for each normal flow segment. Length is NBOU."""
        return self._IBTYPE

    @IBTYPE.setter
    def IBTYPE(self, value: list[int]):
        self._IBTYPE = value

    @property
    def NBVV(self) -> list[numpy.ndarray]:
        """list[numpy.ndarray]: The ADCIRC node indices for each normal flow
        boundary segment. Length is NBOU. Each array hase shape (NVELL[k],).
        """
        return self._NBVV

    @NBVV.setter
    def NBVV(self, value: list[numpy.ndarray]):
        self._NBVV = value

    @property
    def BARLANHT(self) -> list[numpy.ndarray]:
        """numpy.ndarray: Barrier heights along each external normal flow
        boundary segment for each node. Length is NBOU. Each array has shape
        (NVELL[k],).
        """
        return self._BARLANHT

    @BARLANHT.setter
    def BARLANHT(self, value: list[numpy.ndarray]):
        self._BARLANHT = value

    @property
    def BARLANCFSP(self) -> list[numpy.ndarray]:
        """numpy.ndarray: Coefficient for free surface supercritical flow along
        each external normal flow boundary segment for each node. Length is
        NBOU. Each array has shape (NVELL[k],).
        """
        return self._BARLANCFSP

    @BARLANCFSP.setter
    def BARLANCFSP(self, value: list[numpy.ndarray]):
        self._BARLANCFSP = value

    @property
    def IBCONN(self) -> list[numpy.ndarray]:
        """numpy.ndarray: ADCIRC indices of the back face nodes paired with
        front face nodes, along each normal flow boundary segment for each node.
        Length is NBOU. Each array has shape (NVELL[k],).
        """
        return self._IBCONN

    @IBCONN.setter
    def IBCONN(self, value: list[numpy.ndarray]):
        self._IBCONN = value

    @property
    def BARINHT(self) -> list[numpy.ndarray]:
        """numpy.ndarray: Barrier heights along each internal normal flow
        boundary segment for each node. Length is NBOU. Each array has shape
        (NVELL[k],).
        """
        return self._BARINHT

    @BARINHT.setter
    def BARINHT(self, value: list[numpy.ndarray]):
        self._BARINHT = value

    @property
    def BARINCFSB(self) -> list[numpy.ndarray]:
        """numpy.ndarray: Coefficient for free surface subcritical flow along
        each internal normal flow boundary segment for each node. Length is
        NBOU. Each array has shape (NVELL[k],).
        """
        return self._BARINCFSB

    @BARINCFSB.setter
    def BARINCFSB(self, value: list[numpy.ndarray]):
        self._BARINCFSB = value

    @property
    def BARINCFSP(self) -> list[numpy.ndarray]:
        """numpy.ndarray: Coefficient for free surface supercritical flow along
        each internal normal flow boundary segment for each node. Length is
        NBOU. Each array has shape (NVELL[k],).
        """
        return self._BARINCFSP

    @BARINCFSP.setter
    def BARINCFSP(self, value: list[numpy.ndarray]):
        self._BARINCFSP = value

    @property
    def PIPEHT(self) -> list[numpy.ndarray]:
        """numpy.ndarray: Cross barrier pipe height along each internal normal
        flow boundary for each node. Length is NBOU. Each array has shape
        (NVELL[k],).
        """
        return self._PIPEHT

    @PIPEHT.setter
    def PIPEHT(self, value: list[numpy.ndarray]):
        self._PIPEHT = value

    @property
    def PIPECOEF(self) -> list[numpy.ndarray]:
        """numpy.ndarray: Cross barrier pipe bulk friction factor along each
        internal normal flow boundary for each node. Length is NBOU. Each array
        has shape (NVELL[k],).
        """
        return self._PIPECOEF

    @PIPECOEF.setter
    def PIPECOEF(self, value: list[numpy.ndarray]):
        self._PIPECOEF = value

    @property
    def PIPEDIAM(self) -> list[numpy.ndarray]:
        """numpy.ndarray: Cross barrier pipe diameter along each internal normal
        flow boundary for each node. Length is NBOU. Each array has shape
        (NVELL[k],).
        """
        return self._PIPEDIAM

    @PIPEDIAM.setter
    def PIPEDIAM(self, value: list[numpy.ndarray]):
        self._PIPEDIAM = value

    @classmethod
    def from_fort_14(cls, filepath: str) -> 'ADCIRCMesh':
        """Instantiate an ADCIRCMesh object from a fort.14 ADCIRC mesh file.

        Arguments:
            filepath (str): Path to a fort.14 file.

        Returns:
            ADCIRCMesh: A representation of the ADCIRC mesh data.
        """
        with open(filepath, 'r') as f:
            # Read name
            AGRID = f.readline().strip()

            # Read number of elements and nodes
            NE, NP = numpy.loadtxt(f, dtype=int, max_rows=1, usecols=(0, 1))

            # Read node data
            dat = numpy.loadtxt(f, dtype=float, max_rows=NP,
                usecols=(0, 1, 2, 3))
            JN = dat[:, 0].astype(int) # Node indices
            X = dat[:, 1] # Horizontal coordinate
            Y = dat[:, 2] # Vertical coordinate
            DP = dat[:, 3] # Bathymetry

            # Read element data
            dat = numpy.loadtxt(f, dtype=int, max_rows=NE,
                usecols=(0, 1, 2, 3, 4))
            JE = dat[:, 0] # Element indices
            # Ignore NHY for now
            NM = dat[:, 2:5] # Element connectivity

            # Read number of elevation boundary segments and nodes
            NOPE = numpy.loadtxt(f, dtype=int, max_rows=1, usecols=0) # Number of elevation segments
            NETA = numpy.loadtxt(f, dtype=int, max_rows=1, usecols=0) # Number of elevation nodes

            # Read elevation boundary data
            NVDLL = [] # Number of nodes in each elevation segment
            IBTYPEE = [] # Elevation bonudary segment type
            NBDV = [] # Node numbers for each elevation segment
            for k in range(NOPE):
                NVDLLK = numpy.loadtxt(f, dtype=int, max_rows=1, usecols=0)
                IBTYPEEK = 0 # Only accepted value
                NBDVK = numpy.loadtxt(f, dtype=int, max_rows=NVDLLK, usecols=0)
                NVDLL.append(NVDLLK)
                IBTYPEE.append(IBTYPEEK)
                NBDV.append(NBDVK)

            # Read number of normal flow boundary segments and nodes
            NBOU = numpy.loadtxt(f, dtype=int, max_rows=1, usecols=0) # Number of normal flow segments
            NVEL = numpy.loadtxt(f, dtype=int, max_rows=1, usecols=0) # Number of normal flow nodes

            # Read velocity boundary data
            NVELL = [] # Number of nodes in each segment
            IBTYPE = [] # Velocity boundary segment type
            NBVV = [] # Node numbers for each elevation segment
            # Additional data for different boundary types
            BARLANHT = []
            BARLANCFSP = []
            IBCONN = []
            BARINHT = []
            BARINCFSB = []
            BARINCFSP = []
            PIPEHT = []
            PIPECOEF = []
            PIPEDIAM = []
            for k in range(NBOU):
                NVELLK, IBTYPEK = numpy.loadtxt(f, dtype=int, max_rows=1,
                    usecols=(0, 1))
                NBVVK = None
                BARLANHTK = None
                BARLANCFSPK = None
                IBCONNK = None
                BARINHTK = None
                BARINCFSBK = None
                BARINCFSPK = None
                PIPEHTK = None
                PIPECOEFK = None
                PIPEDIAMK = None
                if IBTYPEK in [0, 1, 2, 10, 11, 12, 20, 21, 22, 30]:
                    NBVVK = numpy.loadtxt(f, dtype=int, max_rows=NVELLK, usecols=0)
                elif IBTYPEK in [3, 13, 23]:
                    dat = numpy.loadtxt(f, dtype=float, max_rows=NVELLK,
                        usecols=(0, 1, 2))
                    NBVVK = dat[:, 0].astype(int)
                    BARLANHTK = dat[:, 1]
                    BARLANCFSPK = dat[:, 2]
                elif IBTYPEK in [4, 24]:
                    dat = numpy.loadtxt(f, dtype=float, max_rows=NVELLK,
                        usecols=(0, 1, 2, 3, 4))
                    NBVVK = dat[:, 0].astype(int)
                    IBCONNK = dat[:, 1].astype(int)
                    BARINHTK = dat[:, 2]
                    BARINCFSBK = dat[:, 3]
                    BARINCFSPK = dat[:, 4]
                elif IBTYPEK in [5, 25]:
                    dat = numpy.loadtxt(f, dtype=float, max_rows=NVELLK,
                        usecols=(0, 1, 2, 3, 4, 5, 6, 7))
                    NBVVK = dat[:, 0].astype(int)
                    IBCONNK = dat[:, 1].astype(int)
                    BARINHTK = dat[:, 2]
                    BARINCFSBK = dat[:, 3]
                    BARINCFSPK = dat[:, 4]
                    PIPEHTK = dat[:, 5]
                    PIPECOEFK = dat[:, 6]
                    PIPEDIAMK = dat[:, 7]
                NVELL.append(NVELLK)
                IBTYPE.append(IBTYPEK)
                NBVV.append(NBVVK)
                BARLANHT.append(BARLANHTK)
                BARLANCFSP.append(BARLANCFSPK)
                IBCONN.append(IBCONNK)
                BARINHT.append(BARINHTK)
                BARINCFSB.append(BARINCFSBK)
                BARINCFSP.append(BARINCFSPK)
                PIPEHT.append(PIPEHTK)
                PIPECOEF.append(PIPECOEFK)
                PIPEDIAM.append(PIPEDIAMK)
        return cls(
            AGRID = AGRID,
            NE = NE,
            NP = NP,
            JN = JN,
            X = X,
            Y = Y,
            DP = DP,
            JE = JE,
            NM = NM,
            NOPE = NOPE,
            NETA = NETA,
            NVDLL = NVDLL,
            IBTYPEE = IBTYPEE,
            NBDV = NBDV,
            NBOU = NBOU,
            NVEL = NVEL,
            NVELL = NVELL,
            IBTYPE = IBTYPE,
            NBVV = NBVV,
            BARLANHT = BARLANHT,
            BARLANCFSP = BARLANCFSP,
            IBCONN = IBCONN,
            BARINHT = BARINHT,
            BARINCFSB = BARINCFSB,
            BARINCFSP = BARINCFSP,
            PIPEHT = PIPEHT,
            PIPECOEF = PIPECOEF,
            PIPEDIAM = PIPEDIAM
        )

class Mesh:
    """Wrapper for a generic mesh object."""

    def __init__(self, coordinates: numpy.ndarray, elements: numpy.ndarray):
        """Constructor.

        Arguments:
            coordinates (numpy.ndarray): The coordinates of each node.
            elements (numpy.ndarray): The nodes in each element."""
        self.coordinates = coordinates
        self.elements = elements

    @property
    def coordinates(self) -> numpy.ndarray:
        """numpy.ndarray: The coordinates of each node."""
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value: numpy.ndarray):
        self._coordinates = value
        self._num_nodes = value.shape[0]

    @property
    def elements(self) -> numpy.ndarray:
        """numpy.ndarray: The nodes in each element."""
        return self._elements

    @elements.setter
    def elements(self, value: numpy.ndarray):
        self._elements = value
        self._num_elements = value.shape[0]

    @property
    def num_nodes(self) -> int:
        """int: The number of nodes in the mesh."""
        return self._num_nodes

        @property
        def num_elements(self) -> int:
            """int: The number of elements in the mesh."""
            return self._num_elements

    @classmethod
    def from_ADCIRCMesh(cls, adcirc_mesh: ADCIRCMesh) -> 'Mesh':
        """Construct a generic mesh from an ADCIRC mesh.

        Arguments:
            adcirc_mesh (ADCIRCMesh): A complete ADCIRC mesh object which
                follows ADCIRC mesh conventions, including JN and JE being
                monotonically increasing.
        Returns:
            Mesh: A pared-down 2D mesh structure.
        """
        coordinates = numpy.column_stack((adcirc_mesh.X, adcirc_mesh.Y))
        elements = numpy.searchsorted(adcirc_mesh.JN, adcirc_mesh.NM)
        return cls(coordinates=coordinates, elements=elements)

    def filter_by_coordinates(self, coordinate_filter) -> tuple['Mesh',
        numpy.ndarray, numpy.ndarray]:
        """Generate a new mesh by coordinate filter.

        Arguments:
            coordinate_filter: Callable[[float, float], bool]: A filter that
                takes in the 2D coordinates and returns a boolean.

        Returns:
            tuple[Mesh, numpy.ndarray, numpy.ndarray]
            Mesh: A mesh for which all nodes and elements are contained within
                the filter. This should be able to be applied to numpy.ndarrays
                in a vectorized manner.
            numpy.ndarray: A mapping from the old mesh nodes indices to the
                filtered mesh node indices. Filtered nodes are -1.
            numpy.ndarray: A mapping from the old mesh element indices to the
                filtered mesh element indices. Filtered elements are -1.
        """
        # Figure out which nodes are in range
        mask_nodes_in_filter = coordinate_filter(
            self.coordinates[:, 0],
            self.coordinates[:, 1]
        )

        # Figure out which elements are in range,
        # i.e. every node is in range
        mask_elements_in_filter = (
            mask_nodes_in_filter[self.elements[:, 0]] &
            mask_nodes_in_filter[self.elements[:, 1]] &
            mask_nodes_in_filter[self.elements[:, 2]]
        )

        # Figure out which nodes are associated with a filtered element
        # Sometimes there are orphan nodes, whose neighbors are all out of range
        nodes_in_filtered_elements = numpy.unique(
            self.elements[mask_elements_in_filter]
        )
        mask_nodes_in_elements = numpy.isin(
            numpy.arange(self.num_nodes), # Implicit node indices
            nodes_in_filtered_elements
        )

        # Create new coordinates array
        mask_nodes = mask_nodes_in_filter & mask_nodes_in_elements
        filtered_coordinates = self.coordinates[mask_nodes]

        # Create new elements array
        # Indices should also be renumbered appropriately
        filtered_elements = self.elements[mask_elements_in_filter]
        node_indices = numpy.cumsum(mask_nodes) - 1 # Renumbering
        node_indices[~mask_nodes] = -1
        filtered_elements = node_indices[filtered_elements]
        element_indices = numpy.cumsum(mask_elements_in_filter) # Renumbering
        element_indices[~mask_elements_in_filter] = -1

        # Construct a new object from the result
        return (
            type(self)(filtered_coordinates, filtered_elements),
            node_indices,
            element_indices
        )
