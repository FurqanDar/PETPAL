"""Label maps stored as dictionaries.

This module contains label maps for use in running PETPAL.
"""
from collections.abc import MutableSequence, Callable
import pathlib
from numbers import Integral
from ..utils.image_io import safe_load_meta
from ..utils.useful_functions import str_to_camel_case
from ..utils.image_io import read_label_map_tsv

label_map_freesurfer = {
    'Unknown': 0,
    'LeftCerebralWhiteMatter': 2,
    'LeftLateralVentricle': 4,
    'LeftInfLatVent': 5,
    'LeftCerebellumWhiteMatter': 7,
    'LeftCerebellumCortex': 8,
    'LeftThalamus': 10,
    'LeftCaudate': 11,
    'LeftPutamen': 12,
    'LeftPallidum': 13,
    '3rdVentricle': 14,
    '4thVentricle': 15,
    'BrainStem': 16,
    'LeftHippocampus': 17,
    'LeftAmygdala': 18,
    'CSF': 24,
    'LeftAccumbensArea': 26,
    'LeftVentralDC': 28,
    'LeftVessel': 30,
    'LeftChoroidPlexus': 31,
    'RightCerebralWhiteMatter': 41,
    'RightLateralVentricle': 43,
    'RightInfLatVent': 44,
    'RightCerebellumWhiteMatter': 46,
    'RightCerebellumCortex': 47,
    'RightThalamus': 49,
    'RightCaudate': 50,
    'RightPutamen': 51,
    'RightPallidum': 52,
    'RightHippocampus': 53,
    'RightAmygdala': 54,
    'RightAccumbensArea': 58,
    'RightVentralDC': 60,
    'RightVessel': 62,
    'RightChoroidPlexus': 63,
    'OpticChiasm': 85,
    'AirCavity': 130,
    'Skull': 165,
    'Vermis': 172,
    'Midbrain': 173,
    'Pons': 174,
    'Medulla': 175,
    'CCPosterior': 251,
    'CCMidPosterior': 252,
    'CCCentral': 253,
    'CCMidAnterior': 254,
    'CCAnterior': 255,
    'CSFExtraCerebral': 257,
    'HeadExtraCerebral': 258,
    'CtxLhBankssts': 1001,
    'CtxLhCaudalanteriorcingulate': 1002,
    'CtxLhCaudalmiddlefrontal': 1003,
    'CtxLhCuneus': 1005,
    'CtxLhEntorhinal': 1006,
    'CtxLhFusiform': 1007,
    'CtxLhInferiorparietal': 1008,
    'CtxLhInferiortemporal': 1009,
    'CtxLhIsthmuscingulate': 1010,
    'CtxLhLateraloccipital': 1011,
    'CtxLhLateralorbitofrontal': 1012,
    'CtxLhLingual': 1013,
    'CtxLhMedialorbitofrontal': 1014,
    'CtxLhMiddletemporal': 1015,
    'CtxLhParahippocampal': 1016,
    'CtxLhParacentral': 1017,
    'CtxLhParsopercularis': 1018,
    'CtxLhParsorbitalis': 1019,
    'CtxLhParstriangularis': 1020,
    'CtxLhPericalcarine': 1021,
    'CtxLhPostcentral': 1022,
    'CtxLhPosteriorcingulate': 1023,
    'CtxLhPrecentral': 1024,
    'CtxLhPrecuneus': 1025,
    'CtxLhRostralanteriorcingulate': 1026,
    'CtxLhRostralmiddlefrontal': 1027,
    'CtxLhSuperiorfrontal': 1028,
    'CtxLhSuperiorparietal': 1029,
    'CtxLhSuperiortemporal': 1030,
    'CtxLhSupramarginal': 1031,
    'CtxLhFrontalpole': 1032,
    'CtxLhTemporalpole': 1033,
    'CtxLhTransversetemporal': 1034,
    'CtxLhInsula': 1035,
    'CtxRhBankssts': 2001,
    'CtxRhCaudalanteriorcingulate': 2002,
    'CtxRhCaudalmiddlefrontal': 2003,
    'CtxRhCuneus': 2005,
    'CtxRhEntorhinal': 2006,
    'CtxRhFusiform': 2007,
    'CtxRhInferiorparietal': 2008,
    'CtxRhInferiortemporal': 2009,
    'CtxRhIsthmuscingulate': 2010,
    'CtxRhLateraloccipital': 2011,
    'CtxRhLateralorbitofrontal': 2012,
    'CtxRhLingual': 2013,
    'CtxRhMedialorbitofrontal': 2014,
    'CtxRhMiddletemporal': 2015,
    'CtxRhParahippocampal': 2016,
    'CtxRhParacentral': 2017,
    'CtxRhParsopercularis': 2018,
    'CtxRhParsorbitalis': 2019,
    'CtxRhParstriangularis': 2020,
    'CtxRhPericalcarine': 2021,
    'CtxRhPostcentral': 2022,
    'CtxRhPosteriorcingulate': 2023,
    'CtxRhPrecentral': 2024,
    'CtxRhPrecuneus': 2025,
    'CtxRhRostralanteriorcingulate': 2026,
    'CtxRhRostralmiddlefrontal': 2027,
    'CtxRhSuperiorfrontal': 2028,
    'CtxRhSuperiorparietal': 2029,
    'CtxRhSuperiortemporal': 2030,
    'CtxRhSupramarginal': 2031,
    'CtxRhFrontalpole': 2032,
    'CtxRhTemporalpole': 2033,
    'CtxRhTransversetemporal': 2034,
    'CtxRhInsula': 2035
    }


label_map_freesurfer_merge_lr = {
    'Unknown': 0,
    'CerebralWhiteMatter': [2, 41],
    'LateralVentricle': [4, 43],
    'InfLatVent': [5, 44],
    'CerebellumWhiteMatter': [7, 46],
    'CerebellumCortex': [8, 47],
    'Thalamus': [10, 49],
    'Caudate': [11, 50],
    'Putamen': [12, 51],
    'Pallidum': [13, 52],
    '3rdVentricle': 14,
    '4thVentricle': 15,
    'BrainStem': 16,
    'Hippocampus': [17, 53],
    'Amygdala': [18, 54],
    'CSF': 24,
    'AccumbensArea': [26, 58],
    'VentralDC': [28, 60],
    'Vessel': [30, 62],
    'ChoroidPlexus': [31, 63],
    'OpticChiasm': 85,
    'AirCavity': 130,
    'Skull': 165,
    'Vermis': 172,
    'Midbrain': 173,
    'Pons': 174,
    'Medulla': 175,
    'CCPosterior': 251,
    'CCMidPosterior': 252,
    'CCCentral': 253,
    'CCMidAnterior': 254,
    'CCAnterior': 255,
    'CSFExtraCerebral': 257,
    'HeadExtraCerebral': 258,
    'CtxBankssts': [1001, 2001],
    'CtxCaudalanteriorcingulate': [1002, 2002],
    'CtxCaudalmiddlefrontal': [1003, 2003],
    'CtxCuneus': [1005, 2005],
    'CtxEntorhinal': [1006, 2006],
    'CtxFusiform': [1007, 2007],
    'CtxInferiorparietal': [1008, 2008],
    'CtxInferiortemporal': [1009, 2009],
    'CtxIsthmuscingulate': [1010, 2010],
    'CtxLateraloccipital': [1011, 2011],
    'CtxLateralorbitofrontal': [1012, 2012],
    'CtxLingual': [1013, 2013],
    'CtxMedialorbitofrontal': [1014, 2014],
    'CtxMiddletemporal': [1015, 2015],
    'CtxParahippocampal': [1016, 2016],
    'CtxParacentral': [1017, 2017],
    'CtxParsopercularis': [1018, 2018],
    'CtxParsorbitalis': [1019, 2019],
    'CtxParstriangularis': [1020, 2020],
    'CtxPericalcarine': [1021, 2021],
    'CtxPostcentral': [1022, 2022],
    'CtxPosteriorcingulate': [1023, 2023],
    'CtxPrecentral': [1024, 2024],
    'CtxPrecuneus': [1025, 2025],
    'CtxRostralanteriorcingulate': [1026, 2026],
    'CtxRostralmiddlefrontal': [1027, 2027],
    'CtxSuperiorfrontal': [1028, 2028],
    'CtxSuperiorparietal': [1029, 2029],
    'CtxSuperiortemporal': [1030, 2030],
    'CtxSupramarginal': [1031, 2031],
    'CtxFrontalpole': [1032, 2032],
    'CtxTemporalpole': [1033, 2033],
    'CtxTransversetemporal': [1034, 2034],
    'CtxInsula': [1035, 2035]
    }


label_map_perlcyno = {
    "LhCorticalGrayMatter": 1002,
    "LhSubcorticalAndCerebellarGrayMatter": 1003,
    "LhWhiteMatter": 1004,
    "LhVessels": 1005,
    "LhLateralAndVentralPallium": 1012,
    "LhHippocampalFormation": 1021,
    "LhFornix": 1025,
    "LhVentropallialAmygdala": 1028,
    "LhLateropallialAmygdala": 1032,
    "LhSubpallialAmygdala": 1042,
    "LhCaudateHead": 1056,
    "LhCaudateTail": 1057,
    "LhPutamen": 1058,
    "LhAmygdalostriatalTransition": 1059,
    "LhInternalCapsule": 1060,
    "LhAccumbens": 1062,
    "LhOlfactoryTubercule": 1063,
    "LhExternalGlobusPallidus": 1066,
    "LhInternalGlobusPallidus": 1067,
    "LhAnsaLenticularis": 1068,
    "LhAnteriorCommissure": 1069,
    "LhVentralPallidum": 1070,
    "LhBasalNucleusRegion": 1072,
    "LhBedNucleusOfTheStriaTerminalis": 1075,
    "LhSeptumDiagonalBandComplex": 1076,
    "LhPreopticComplex": 1079,
    "LhTuberalHypothalamus": 1086,
    "LhPosteriorHypothalamus": 1102,
    "LhPituitary": 1110,
    "LhZonaIncertaAndLenticularFascicles": 1114,
    "LhReticularThalamus": 1116,
    "LhMedialThalamus": 1117,
    "LhMidlineThalamus": 1130,
    "LhVentralThalamus": 1138,
    "LhPosteriorThalamus": 1151,
    "LhGeniculateThalamus": 1160,
    "LhEpithalamus": 1163,
    "LhPretectum": 1166,
    "LhSuperiorColliculus": 1176,
    "LhInferiorColliculusComplex": 1177,
    "LhPeriaqueductalGrayRegion": 1180,
    "LhLateralMidbrain": 1183,
    "LhMedialMidbrain": 1195,
    "LhRubralRegion": 1204,
    "LhMidbrainDopaminergicComplex": 1207,
    "LhDorsalPons": 1216,
    "LhLateralPons": 1229,
    "LhPontineReticulum": 1235,
    "LhSuperiorOliveLemiscalRegion": 1241,
    "LhPontineNucleus": 1246,
    "LhPontineFibers": 1247,
    "LhCerebellarWhiteMatter": 1252,
    "LhDeepCerebellarNuclei": 1255,
    "LhFlocculus": 1263,
    "LhParaflocculus": 1264,
    "LhAnteriorVermisCerebellarCortex": 1266,
    "LhPosteriorVermisCerebellarCortex": 1272,
    "LhIntermediateCerebellarCortex": 1278,
    "LhLateralCerebellarCortex": 1286,
    "LhVestibuloCochlearComplex": 1292,
    "LhTrigeminalComplex": 1297,
    "LhSomatovisceralComplex": 1304,
    "LhIntermediateMedulla": 1309,
    "LhVentralMedulla": 1321,
    "LhAnteriorCingulateCortex": 1353,
    "LhMidcingulateCortex": 1361,
    "LhMedialOrbitalFrontalCortex": 1367,
    "LhLateralOrbitalFrontalCortex": 1375,
    "LhCaudalOrbitalFrontalCortex": 1387,
    "LhFrontalEyeFieldsPeriarcuateArea8a": 1401,
    "LhDorsolateralPrefrontalCortex": 1404,
    "LhVentrolateralPrefrontalCortex": 1414,
    "LhPrimaryMotorCortex": 1429,
    "LhDorsalPremotorCortex": 1431,
    "LhVentralPremotorCortex": 1434,
    "LhMedialSupplementaryMotorAreas": 1437,
    "LhPrimarySomatosensoryCortex": 1442,
    "LhSecondarySomatosensoryCortex": 1445,
    "LhSuperiorParietalLobule": 1446,
    "LhInferiorParietalLobule": 1462,
    "LhPosteriorMedialCortex": 1475,
    "LhParahippocampalCortex": 1498,
    "LhRhinalCortex": 1503,
    "LhTemporalPole": 1515,
    "LhAreaTEO": 1526,
    "LhAreaTE": 1527,
    "LhFundusOfTheSuperiorTemporalSulcus": 1538,
    "LhRostralSuperiorTemporalRegion": 1544,
    "LhCaudalSuperiorTemporalGyrus": 1549,
    "LhBeltAreasOfAuditoryCortex": 1555,
    "LhPolarRostrotemporalCortex": 1568,
    "LhCoreAreasOfAuditoryCortex": 1569,
    "LhFloorOfTheLateralSulcus": 1574,
    "LhMiddleTemporalArea": 1582,
    "LhVisualArea4": 1584,
    "LhVisualArea3": 1588,
    "LhVisualArea2": 1593,
    "LhPrimaryVisualCortex": 1596,
    "RhCorticalGrayMatter": 2002,
    "RhSubcorticalAndCerebellarGrayMatter": 2003,
    "RhWhiteMatter": 2004,
    "RhVessels": 2005,
    "RhLateralAndVentralPallium": 2012,
    "RhHippocampalFormation": 2021,
    "RhFornix": 2025,
    "RhVentropallialAmygdala": 2028,
    "RhLateropallialAmygdala": 2032,
    "RhSubpallialAmygdala": 2042,
    "RhCaudateHead": 2056,
    "RhCaudateTail": 2057,
    "RhPutamen": 2058,
    "RhAmygdalostriatalTransition": 2059,
    "RhInternalCapsule": 2060,
    "RhAccumbens": 2062,
    "RhOlfactoryTubercule": 2063,
    "RhExternalGlobusPallidus": 2066,
    "RhInternalGlobusPallidus": 2067,
    "RhAnsaLenticularis": 2068,
    "RhAnteriorCommissure": 2069,
    "RhVentralPallidum": 2070,
    "RhBasalNucleusRegion": 2072,
    "RhBedNucleusOfTheStriaTerminalis": 2075,
    "RhSeptumDiagonalBandComplex": 2076,
    "RhPreopticComplex": 2079,
    "RhTuberalHypothalamus": 2086,
    "RhPosteriorHypothalamus": 2102,
    "RhPituitary": 2110,
    "RhZonaIncertaAndLenticularFascicles": 2114,
    "RhReticularThalamus": 2116,
    "RhMedialThalamus": 2117,
    "RhMidlineThalamus": 2130,
    "RhVentralThalamus": 2138,
    "RhPosteriorThalamus": 2151,
    "RhGeniculateThalamus": 2160,
    "RhEpithalamus": 2163,
    "RhPretectum": 2166,
    "RhSuperiorColliculus": 2176,
    "RhInferiorColliculusComplex": 2177,
    "RhPeriaqueductalGrayRegion": 2180,
    "RhLateralMidbrain": 2183,
    "RhMedialMidbrain": 2195,
    "RhRubralRegion": 2204,
    "RhMidbrainDopaminergicComplex": 2207,
    "RhDorsalPons": 2216,
    "RhLateralPons": 2229,
    "RhPontineReticulum": 2235,
    "RhSuperiorOliveLemiscalRegion": 2241,
    "RhPontineNucleus": 2246,
    "RhPontineFibers": 2247,
    "RhCerebellarWhiteMatter": 2252,
    "RhDeepCerebellarNuclei": 2255,
    "RhFlocculus": 2263,
    "RhParaflocculus": 2264,
    "RhAnteriorVermisCerebellarCortex": 2266,
    "RhPosteriorVermisCerebellarCortex": 2272,
    "RhIntermediateCerebellarCortex": 2278,
    "RhLateralCerebellarCortex": 2286,
    "RhVestibuloCochlearComplex": 2292,
    "RhTrigeminalComplex": 2297,
    "RhSomatovisceralComplex": 2304,
    "RhIntermediateMedulla": 2309,
    "RhVentralMedulla": 2321,
    "RhAnteriorCingulateCortex": 2353,
    "RhMidcingulateCortex": 2361,
    "RhMedialOrbitalFrontalCortex": 2367,
    "RhLateralOrbitalFrontalCortex": 2375,
    "RhCaudalOrbitalFrontalCortex": 2387,
    "RhFrontalEyeFieldsPeriarcuateArea8a": 2401,
    "RhDorsolateralPrefrontalCortex": 2404,
    "RhVentrolateralPrefrontalCortex": 2414,
    "RhPrimaryMotorCortex": 2429,
    "RhDorsalPremotorCortex": 2431,
    "RhVentralPremotorCortex": 2434,
    "RhMedialSupplementaryMotorAreas": 2437,
    "RhPrimarySomatosensoryCortex": 2442,
    "RhSecondarySomatosensoryCortex": 2445,
    "RhSuperiorParietalLobule": 2446,
    "RhInferiorParietalLobule": 2462,
    "RhPosteriorMedialCortex": 2475,
    "RhParahippocampalCortex": 2498,
    "RhRhinalCortex": 2503,
    "RhTemporalPole": 2515,
    "RhAreaTEO": 2526,
    "RhAreaTE": 2527,
    "RhFundusOfTheSuperiorTemporalSulcus": 2538,
    "RhRostralSuperiorTemporalRegion": 2544,
    "RhCaudalSuperiorTemporalGyrus": 2549,
    "RhBeltAreasOfAuditoryCortex": 2555,
    "RhPolarRostrotemporalCortex": 2568,
    "RhCoreAreasOfAuditoryCortex": 2569,
    "RhFloorOfTheLateralSulcus": 2574,
    "RhMiddleTemporalArea": 2582,
    "RhVisualArea4": 2584,
    "RhVisualArea3": 2588,
    "RhVisualArea2": 2593,
    "RhPrimaryVisualCortex": 2596
}


label_map_perlcyno_merge_lr = {
    "CSF": 1,
    "CorticalGrayMatter": 2,
    "SubcorticalAndCerebellarGrayMatter": 3,
    "WhiteMatter": 4,
    "Vessels": 5,
    "LateralAndVentralPallium": 12,
    "HippocampalFormation": 21,
    "Fornix": 25,
    "VentropallialAmygdala": 28,
    "LateropallialAmygdala": 32,
    "SubpallialAmygdala": 42,
    "CaudateHead": 56,
    "CaudateTail": 57,
    "Putamen": 58,
    "AmygdalostriatalTransition": 59,
    "InternalCapsule": 60,
    "Accumbens": 62,
    "OlfactoryTubercule": 63,
    "ExternalGlobusPallidus": 66,
    "InternalGlobusPallidus": 67,
    "AnsaLenticularis": 68,
    "AnteriorCommissure": 69,
    "VentralPallidum": 70,
    "BasalNucleusRegion": 72,
    "BedNucleusOfTheStriaTerminalis": 75,
    "SeptumDiagonalBandComplex": 76,
    "PreopticComplex": 79,
    "TuberalHypothalamus": 86,
    "PosteriorHypothalamus": 102,
    "Pituitary": 110,
    "ZonaIncertaAndLenticularFascicles": 114,
    "ReticularThalamus": 116,
    "MedialThalamus": 117,
    "MidlineThalamus": 130,
    "VentralThalamus": 138,
    "PosteriorThalamus": 151,
    "GeniculateThalamus": 160,
    "Epithalamus": 163,
    "Pretectum": 166,
    "SuperiorColliculus": 176,
    "InferiorColliculusComplex": 177,
    "PeriaqueductalGrayRegion": 180,
    "LateralMidbrain": 183,
    "MedialMidbrain": 195,
    "RubralRegion": 204,
    "MidbrainDopaminergicComplex": 207,
    "DorsalPons": 216,
    "LateralPons": 229,
    "PontineReticulum": 235,
    "SuperiorOliveLemiscalRegion": 241,
    "PontineNucleus": 246,
    "PontineFibers": 247,
    "CerebellarWhiteMatter": 252,
    "DeepCerebellarNuclei": 255,
    "Flocculus": 263,
    "Paraflocculus": 264,
    "AnteriorVermisCerebellarCortex": 266,
    "PosteriorVermisCerebellarCortex": 272,
    "IntermediateCerebellarCortex": 278,
    "LateralCerebellarCortex": 286,
    "VestibuloCochlearComplex": 292,
    "TrigeminalComplex": 297,
    "SomatovisceralComplex": 304,
    "IntermediateMedulla": 309,
    "VentralMedulla": 321,
    "AnteriorCingulateCortex": 353,
    "MidcingulateCortex": 361,
    "MedialOrbitalFrontalCortex": 367,
    "LateralOrbitalFrontalCortex": 375,
    "CaudalOrbitalFrontalCortex": 387,
    "FrontalEyeFieldsPeriarcuateArea8a": 401,
    "DorsolateralPrefrontalCortex": 404,
    "VentrolateralPrefrontalCortex": 414,
    "PrimaryMotorCortex": 429,
    "DorsalPremotorCortex": 431,
    "VentralPremotorCortex": 434,
    "MedialSupplementaryMotorAreas": 437,
    "PrimarySomatosensoryCortex": 442,
    "SecondarySomatosensoryCortex": 445,
    "SuperiorParietalLobule": 446,
    "InferiorParietalLobule": 462,
    "PosteriorMedialCortex": 475,
    "ParahippocampalCortex": 498,
    "RhinalCortex": 503,
    "TemporalPole": 515,
    "AreaTEO": 526,
    "AreaTE": 527,
    "FundusOfTheSuperiorTemporalSulcus": 538,
    "RostralSuperiorTemporalRegion": 544,
    "CaudalSuperiorTemporalGyrus": 549,
    "BeltAreasOfAuditoryCortex": 555,
    "PolarRostrotemporalCortex": 568,
    "CoreAreasOfAuditoryCortex": 569,
    "FloorOfTheLateralSulcus": 574,
    "MiddleTemporalArea": 582,
    "VisualArea4": 584,
    "VisualArea3": 588,
    "VisualArea2": 593,
    "PrimaryVisualCortex": 596
}


class LabelMapLoader:
    """Class for loading label map data to convert between segmentation mappings and region
    labels.
    
    There are three ways to load label maps, using argument `label_map_option`:

    1. (str) Load a preset option implemented in PETPAL. These include `freesurfer`,
       `freesurfer_merge_lr`, `perlcyno`, `perlcyno_merge_lr`. See
       :meth:`~LabelMapLoader.from_petpal`.
    #. (str) Load a json file that maps regions used in your study. Provide the path to the
       .json file. See :meth:`~LabelMapLoader.from_json`.
    #. (dict) Load a Python dictionary that implements a label map for region in your study. See
       :meth:`~LabelMapLoader.from_dict`.

    :ivar loader_method: A string or dictionary to load as a label map.
    :ivar label_map: The label map to use in PETPAL methods.
    """
    def __init__(self, label_map_option: str | dict):
        self.loader_method = self.detect_option(label_map_option=label_map_option)
        self.label_map = self.loader_method(label_map_option)
        self.labels_to_camel_case()
        self.validate_mappings()

    def from_petpal(self, label_map_name: str) -> dict:
        """Loads a label map based on an existing list of label maps implemented in PETPAL.
        
        Options include:

        * `freesurfer`: The regions corresponding to the aparc+aseg segmentation image.
        * `freesurfer_merge_lr`: The regions corresponding to the aparc+aseg segmentation image.
          Unilateral map combining each left/right split into one region.
        * `perlcyno`: Bilateral regions in the PerlCyno primate atlas.
        * `perlcyno_merge_lr`: Unilateral map combining left/right regions in the PerlCyno
          primate atlas.
            
        Args:
            label_map_name (str): A name matching a preset option for a label map.
            
        Returns:
            label_map (dict): The label map selected."""
        match label_map_name.lower():
            case 'freesurfer':
                return label_map_freesurfer
            case 'freesurfer_merge_lr':
                return label_map_freesurfer_merge_lr
            case 'perlcyno':
                return label_map_perlcyno
            case 'perlcyno_merge_lr':
                return label_map_perlcyno_merge_lr
            case _:
                raise ValueError(f"Label map name {label_map_name} not in existing list of "
                                 "implemented label maps. Choose one of: 'freesurfer', "
                                 "'freesurfer_merge_lr', 'perlcyno', or 'perlcyno_merge_lr.")

    def from_dict(self, label_map: dict) -> dict:
        """Provide a label map implemented in Python.
        
        Args:
            label_map (dict): The label map implemented for use in the PET study.
            
        Returns:
            label_map (dict): The label map selected."""
        return label_map

    def from_json(self, label_map_path: str) -> dict:
        """Load a label map from a .json file.
        
        Args:
            label_map_path (str): Path to the label map for use in the PET study.
            
        Returns:
            label_map (dict): The label map loaded from file."""
        return safe_load_meta(input_metadata_file=label_map_path)

    def from_dseg_tsv(self, label_map_path: str) -> dict:
        r"""
        Load a label map from a .tsv file.

        Args:
            label_map_path (str): Path to the label map for use in the PET study.

        Returns:
            label_map (dict): The label map loaded from file.
        """
        label_map_df = read_label_map_tsv(label_map_file=label_map_path)
        out_map_dict = {str_to_camel_case(seg): val for seg, val in
                        zip(label_map_df['abbreviation'], label_map_df['mapping'])}

        return out_map_dict


    def detect_option(self, label_map_option: dict | str) -> Callable:
        """Determine the label map loading method to use based on the provided option.
        
        Raises:
            FileNotFoundError: If the label_map_option looks like a path but doesn't point to an
                existing file.
            TypeError: If the label_map_option is not a str or dict instance."""
        if isinstance(label_map_option, dict):
            return self.from_dict
        if isinstance(label_map_option, str):
            label_map_path = pathlib.Path(label_map_option)
            if label_map_path.exists():
                if label_map_path.suffix == '.json':
                    return self.from_json
                elif label_map_path.suffix == '.tsv':
                    return self.from_dseg_tsv
            if label_map_path.suffix!='':
                raise FileNotFoundError(f'Label map option {label_map_option} looks like a path'
                                        'yet does not exist.')
            return self.from_petpal
        raise TypeError(f'label_map_option should be a str or dict. Got type: '
                         f'{type(label_map_option)}')

    def labels_to_camel_case(self):
        """Convert all label map labels to camel case and update label map."""
        label_map = self.label_map.copy()
        labels = label_map.keys()
        for label in labels:
            updated_label = str_to_camel_case(label)
            self.label_map[updated_label] = self.label_map.pop(label)

    def validate_mappings(self):
        """Validate mapping values for integer mappings in the label map. Mappings can be an
        integer or a list of integers.
        
        Raises:
            TypeError: If one of the mappings is not an integer or a list of integers.
        """
        label_map = self.label_map.copy()
        labels = label_map.keys()
        mappings = label_map.values()
        for label, mapping in zip(labels, mappings):
            if isinstance(mapping, Integral):
                continue
            if isinstance(mapping, MutableSequence):
                for value in mapping:
                    if isinstance(value, Integral):
                        continue
                    raise TypeError(f'Label {label} with mapping {mapping} contains value {value} '
                                    f'which is not an integer. Instead found type: {type(value)}.')
            else:
                raise TypeError(f'Label {label} contains mapping {mapping} which is not '
                                f'an integer or a list. Instead found type: {type(mapping)}.')
