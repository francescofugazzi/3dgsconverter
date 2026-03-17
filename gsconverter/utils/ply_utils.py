"""
Minimal PLY binary reader/writer.

Drop-in replacement for the subset of plyfile used by gsconverter.
Supports reading and writing binary little-endian PLY files with scalar
properties (no list properties, no ASCII/big-endian).

Licensed under the MIT License — same as the rest of gsconverter.
"""

import numpy as np
import struct

# ── type mappings ──────────────────────────────────────────────────────────

_PLY_TO_NUMPY = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}

_NUMPY_TO_PLY = {
    "i1": "char",
    "u1": "uchar",
    "i2": "short",
    "u2": "ushort",
    "i4": "int",
    "u4": "uint",
    "f4": "float",
    "f8": "double",
}


def _numpy_kind_to_ply(dtype: np.dtype) -> str:
    """Map a single-field numpy dtype to its PLY type name."""
    key = dtype.byteorder.replace("=", "<").replace("|", "") + dtype.kind + str(dtype.itemsize)
    # Normalise to little-endian / native single-char codes
    short = dtype.kind + str(dtype.itemsize)
    if short in _NUMPY_TO_PLY:
        return _NUMPY_TO_PLY[short]
    raise ValueError(f"Unsupported numpy dtype for PLY: {dtype}")


# ── PlyProperty ────────────────────────────────────────────────────────────

class PlyProperty:
    """Lightweight stand-in so that ``element.properties`` works."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"PlyProperty({self.name!r})"


# ── PlyElement ─────────────────────────────────────────────────────────────

class PlyElement:
    """One named element inside a PLY file (e.g. 'vertex', 'chunk')."""

    __slots__ = ("name", "data", "properties")

    def __init__(self, name: str, data: np.ndarray):
        self.name = name
        self.data = data
        self.properties = [PlyProperty(n) for n in data.dtype.names] if data.dtype.names else []

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"PlyElement({self.name!r}, count={len(self.data)})"

    # ── factory ────────────────────────────────────────────────────────────
    @staticmethod
    def describe(data: np.ndarray, name: str) -> "PlyElement":
        """Create a *PlyElement* from a numpy structured array (mirrors plyfile API)."""
        return PlyElement(name, data)


# ── PlyData ────────────────────────────────────────────────────────────────

class PlyData:
    """Container for a whole PLY file (header + elements)."""

    def __init__(self, elements, text=False, byte_order="<"):
        self.elements = list(elements)
        self.byte_order = byte_order
        self._index = {el.name: el for el in self.elements}

    # ── dict-like access ───────────────────────────────────────────────────
    def __getitem__(self, name: str) -> PlyElement:
        return self._index[name]

    def __contains__(self, name: str) -> bool:
        return name in self._index

    # ── read ───────────────────────────────────────────────────────────────
    @staticmethod
    def read(path: str) -> "PlyData":
        with open(path, "rb") as fh:
            header_lines, header_len = _read_header(fh)

        element_specs = _parse_header(header_lines)

        elements = []
        with open(path, "rb") as fh:
            fh.seek(header_len)
            for name, props in element_specs:
                count, dtype = props
                raw = fh.read(count * dtype.itemsize)
                if len(raw) < count * dtype.itemsize:
                    raise ValueError(
                        f"Unexpected EOF reading element '{name}': "
                        f"expected {count * dtype.itemsize} bytes, got {len(raw)}"
                    )
                arr = np.frombuffer(raw, dtype=dtype, count=count)
                elements.append(PlyElement(name, arr))

        return PlyData(elements, byte_order="<")

    # ── write ──────────────────────────────────────────────────────────────
    def write(self, path: str) -> None:
        lines = ["ply", "format binary_little_endian 1.0"]
        for el in self.elements:
            lines.append(f"element {el.name} {len(el.data)}")
            dt = el.data.dtype
            for field_name in dt.names:
                ply_type = _numpy_kind_to_ply(dt[field_name])
                lines.append(f"property {ply_type} {field_name}")
        lines.append("end_header")

        header_bytes = ("\n".join(lines) + "\n").encode("ascii")

        with open(path, "wb") as fh:
            fh.write(header_bytes)
            for el in self.elements:
                data = el.data
                if data.dtype.byteorder not in ("<", "=", "|"):
                    data = data.byteswap().newbyteorder("<")
                fh.write(data.tobytes())


# ── internal helpers ───────────────────────────────────────────────────────

def _read_header(fh):
    """Return (list_of_header_lines, byte_offset_of_data)."""
    lines = []
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PLY header")
        decoded = line.decode("ascii", errors="ignore").strip()
        lines.append(decoded)
        if decoded == "end_header":
            break
    return lines, fh.tell()


def _parse_header(lines):
    """Parse header lines into a list of (element_name, (count, numpy_dtype))."""
    if not lines or not lines[0].startswith("ply"):
        raise ValueError("Not a PLY file")

    fmt_line = lines[1] if len(lines) > 1 else ""
    if "binary_little_endian" not in fmt_line:
        raise ValueError(
            f"Only binary_little_endian PLY is supported. Got: {fmt_line!r}"
        )

    elements = []          # [(name, (count, dtype))]
    current_name = None
    current_count = 0
    current_props = []     # [(field_name, numpy_type_str)]

    def _flush():
        if current_name is not None:
            dt = np.dtype([(n, f"<{t}") for n, t in current_props])
            elements.append((current_name, (current_count, dt)))

    for line in lines[2:]:
        if line.startswith("element"):
            _flush()
            parts = line.split()
            current_name = parts[1]
            current_count = int(parts[2])
            current_props = []
        elif line.startswith("property"):
            parts = line.split()
            if parts[1] == "list":
                raise ValueError("List properties are not supported")
            ply_type = parts[1]
            prop_name = parts[2]
            np_type = _PLY_TO_NUMPY.get(ply_type)
            if np_type is None:
                raise ValueError(f"Unknown PLY type: {ply_type!r}")
            current_props.append((prop_name, np_type))

    _flush()
    return elements
