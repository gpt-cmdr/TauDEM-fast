# SPDX-License-Identifier: GPL-3.0-only
"""Reads the GeoTIFFs that tests/async_writer.rs keeps in TAUDEM_GDAL_CHECK_DIR
with GDAL and compares their pixels, byte for byte, with the .bin beside each.

    TAUDEM_GDAL_CHECK_DIR=<dir> cargo test --test async_writer
    python tests/gdal_readback.py <dir>
"""

import glob
import os
import sys

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def main(directory):
    tifs = sorted(glob.glob(os.path.join(directory, "*.tif")))
    checked = [t for t in tifs if os.path.exists(t[:-4] + ".bin")]
    if not checked:
        sys.exit(f"no .tif with a .bin in {directory}")
    print(f"GDAL {gdal.__version__}")
    failed = 0
    for tif in checked:
        ds = gdal.Open(tif)
        band = ds.GetRasterBand(1)
        pixels = band.ReadAsArray()
        expected = np.fromfile(tif[:-4] + ".bin", dtype=pixels.dtype)
        same = pixels.tobytes() == expected.tobytes()
        structure = gdal.Info(tif, format="json")
        compression = structure.get("metadata", {}).get("IMAGE_STRUCTURE", {}).get("COMPRESSION")
        checksum = band.Checksum()
        print(
            f"{'ok  ' if same else 'FAIL'} {os.path.basename(tif)}: {ds.RasterXSize}x{ds.RasterYSize} "
            f"{gdal.GetDataTypeName(band.DataType)} blocks {band.GetBlockSize()} {compression} "
            f"nodata {band.GetNoDataValue()} checksum {checksum}"
        )
        failed += not same
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ["TAUDEM_GDAL_CHECK_DIR"])
