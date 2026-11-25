# Large-terrain benchmark candidates (HEC-RAS terrains)

Surveyed 2026-11-25 with `gdalinfo`. Unless noted: Float32, nodata -9999,
256x256 tiles, DEFLATE, internal overviews. Copy to local disk before timing to
keep network I/O out of the measurement.

| Tier | Path | Size | W x H (cells) | Notes |
|---|---|---|---|---|
| 1e7 | `G:\GH\ras-commander\example_projects\Muncie\Terrain\Terrain.muncie_clip.tif` | 17.9 MB | 7892 x 4538 (3.6e7) | Public HEC-RAS example; 5 ft; EPSG:2965 |
| 1e7 | `G:\GH\ras-agent\workspace\Spring Creek Springfield IL\08_model_validation\ras_agent_95mi2\terrain.tif` | 116 MB | 10323 x 5380 (5.6e7) | LZW, striped, no overviews; EPSG:5070. Existing TauDEM outputs in `...\04_terrain\taudem_work\` |
| 1e7-1e8 | `G:\GH\ras-commander\example_projects\BaldEagleCrkMulti2D\Terrain\Terrain50.baldeagledem.tif` | 178 MB | 8643 x 6902 (6.0e7) | Public HEC-RAS example |
| 1e8 | `G:\GH\ras-agent\data\RAS_7.0_Template\Terrain\Terrain.dem_mosaic_5070_basin_extent.tif` | 150 MB | 11630 x 7158 (8.3e7) | 3 m; EPSG:5070 |
| 1e9 | `T:\FEMA eBFE\Organized\LakeMaurepas_08070204\RAS Model\Hydraulics\Terrain\Terrain.LakeMaurepasDEM.tif` | 758 MB | 27477 x 22133 (6.1e8) | ~9.84 ft |
| 1e9 | `G:\GH\ras-commander-docs\working\spring-river-source-local-20260718\Terrain\Terrain (2).ep_extend.tif` | 1.0 GB | 28916 x 26316 (7.6e8) | 10 ft |
| 1e9 | `C:\Temp\rbfs-east-qc-20261004\Terrain\Terrain.cudem_ft.tif` | 517 MB | 27684 x 33398 (9.2e8) | local disk |
| 1e9 | `C:\Temp\ELC_v2_validation\Terrain\2022LiDAR+CoNED.2022LiDAR+CoNED.tif` | 621 MB | 35999 x 31635 (1.14e9) | 3 ft; local disk |
| 1e9+ | `T:\FEMA eBFE\Organized\Amite_08070202\RAS Model\WA1\Terrain\Terrain.DEM_10ft.tif` | 1.86 GB | 25548 x 50272 (1.28e9) | 10 ft |
| 1e9+ | `T:\FEMA eBFE\Organized\Amite_08070202\RAS Model\WA1\Terrain\DEM_10ft.tif` | 5.02 GB | 25000 x 50000 (1.25e9) | Uncompressed source of the above; nodata -3.4e38; I/O contrast |
| 4e9 | `T:\FEMA eBFE\Organized\Tickfaw_08070203\RAS Model\Tickfaw\Terrain\Terrain_Repaired.WS1_Repaired3.tif` | 2.86 GB | 42810 x 93236 (3.99e9) | 1 m; top tier |
| stress | `T:\RBFS\Webmap Extracted\AdamsCowBayou\terrain\AdamsCowBayou_terrain_cog.tif` | 5.46 GB | 480509 x 316295 (1.5e11) | Int16 web COG; stress only |

Unusable: `T:\FEMA eBFE\12060203_Bosque_TX_BLE\Terrain\Terrain.Bosque1.tif`
(truncated); RBFS SpringCreek/UpperSanMarcos COGs (codec missing in local GDAL,
likely ZSTD/LERC — a useful terrio codec test).
