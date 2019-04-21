###       /bin/bash runTestCases_docker.sh

! snow properties
! ====================================================================
! snow albedo
! ====================================================================
!albedoMax                 |       0.8400 |       0.7000 |       0.9500 $$$$$$ 2: 0.75 | 0.92
albedoMinWinter           |       0.5500 |       0.6000 |       1.0000 $$$$$$ 2: 0.65 | 0.90
albedoMinSpring           |       0.5500 |       0.3000 |       1.0000 $$$$$$ 2: 0.40 | 0.90
albedoMaxVisible          |       0.9500 |       0.7000 |       0.9500 $$$$$$ 1: 0.9
albedoMinVisible          |       0.7500 |       0.5000 |       0.7500 $$$$$$ 1: 0.7
albedoMaxNearIR           |       0.6500 |       0.5000 |       0.7500 $$$$$$ 2: 0.55 | 0.70
albedoMinNearIR           |       0.3000 |       0.1500 |       0.4500 $$$$$$ 2: 0.20 | 0.43
albedoDecayRate           |       1.0d+6 |       0.1d+6 |       5.0d+6 $$$$$$ 3: 0.8d+6 | 2d+6 | 4.0d+6
albedoSootLoad            |       0.3000 |       0.1000 |       0.5000
albedoRefresh             |       1.0000 |       1.0000 |      10.0000

#snw_crit             | critical mass necessary for albedo refreshment                    | kg m-2     | scalarv | F
#alb_fresh            | fresh snow albedo                                                 | -          | scalarv | F
#alb_dry              | minimum snow albedo during winter                                 | -          | scalarv | F
#alb_wet              | minimum snow albedo during spring                                 | -          | scalarv | F
#alb_decay            | temporal decay factor for snow albedo                             | s-1        | scalarv | F
#alb_scale            | albedo scaling factor                                             | s          | scalarv | F
#soot_load            | albedo decay associated with soot load                            | -          | scalarv | F
! ====================================================================
! turbulent heat fluxes
! ====================================================================
p1z0Snow                    |       0.0010 |       0.0010 |      10.0000  $$$$$$ 3: 0.002 | 0.009 | 0.02
p2z0Soil                    |       0.0100 |       0.0010 |      10.0000  $$$$$$ 3: 0.005 | 0.10  | 2
p3z0Canopy                  |       0.1000 |       0.0010 |      10.0000  ******
p4zpdFraction               |       0.6500 |       0.5000 |       0.8500  ******
p5critRichNumber            |       0.2000 |       0.1000 |       1.0000  $$$$$$ 3: 0.2 | 0.5 | 0.8
p6Louis79_bparam            |       9.4000 |       9.2000 |       9.6000  A_stability function
p7Louis79_cStar             |       5.3000 |       5.1000 |       5.5000  A_stability function
p8Mahrt87_eScale            |       1.0000 |       0.5000 |       2.0000  A_stability function
p9leafExchangeCoeff         |       0.0100 |       0.0010 |       0.1000  ******
p10windReductionParam        |       0.2800 |       0.0000 |       1.0000  $$$$$$ 3: 0.2 | 0.4 | 0.8
! ====================================================================
! water flow through snow
! ====================================================================
p11Fcapil                    |       0.0600 |       0.0100 |       0.1000
p12k_snow                    |       0.0150 |       0.0050 |       0.0500
p13mw_exp                    |       3.0000 |       1.0000 |       5.0000
! ====================================================================
p14snowfrz_scale             |      50.0000 |      10.0000 |    1000.0000 #Freezing curve for snow
p15fixedThermalCond_snow     |       0.3500 |       0.1000 |       1.0000
! ====================================================================
#! new snow density
#! ====================================================================
#newSnowDenMin             |     100.0000 |      50.0000 |     100.0000
#newSnowDenMult            |     100.0000 |      25.0000 |      75.0000
#newSnowDenScal            |       5.0000 |       1.0000 |       5.0000
#constSnowDen              |     100.0000 |      50.0000 |     250.0000
#newSnowDenAdd             |     109.0000 |      80.0000 |     120.0000
#newSnowDenMultTemp        |       6.0000 |       1.0000 |      12.0000
#newSnowDenMultWind        |      26.0000 |      16.0000 |      36.0000
#newSnowDenMultAnd         |       1.0000 |       1.0000 |       3.0000
#newSnowDenBase            |       0.0000 |       0.0000 |       0.0000
#! ====================================================================
#! radiation transfer within snow
#! ====================================================================
#!!!!!!!!!!radExt_snow               |      20.0000 |      20.0000 |      20.0000
#!!!!!!!!!!directScale               |       0.0900 |       0.0000 |       0.5000
#!!!!!!!!!!Frad_direct               |       0.7000 |       0.0000 |       1.0000
#!!!!!!!!!!Frad_vis                  |       0.5000 |       0.0000 |       1.0000
#
#! ====================================================================
#! snow compaction
#! ====================================================================
#densScalGrowth            |       0.0460 |       0.0230 |       0.0920
#tempScalGrowth            |       0.0400 |       0.0200 |       0.0600
#grainGrowthRate           |       2.7d-6 |       1.0d-6 |       5.0d-6
#densScalOvrbdn            |       0.0230 |       0.0115 |       0.0460
#tempScalOvrbdn            |       0.0800 |       0.6000 |       1.0000
#baseViscosity             |       9.0d+5 |       5.0d+5 |       1.5d+6
#! ====================================================================
! ***********************************************************************************************************************
simulStart              '2010-07-01 00:00'    ! (T-01) simulation start time -- must be in single quotes
simulFinsh              '2011-09-30 00:00'    ! (T-02) simulation end time -- must be in single quotes
! ***********************************************************************************************************************
Models we are going to change
#$$$alb_method                      $varDecay        ! (23) choice of albedo representation ($consdecay)
#$$$windPrfile                      $logBelowCanopy  ! (20) choice of wind profile through the canopy ($Exponential)
#$$$astability                      $louisinv        ! (21) choice of stability function  ($standard , &mahrtexp)
#$$$canopySrad                      $CLM_2stream     ! (22) choice of canopy shortwave radiation method ($noah_mp, $UEB_2stream, $NL_scatter, $BeersLaw)  
#$$$thCondSnow                      $jrdn1991        ! (26) choice of thermal conductivity representation for snow ($tyen1965, $melr1977, $smnv2000)
#$$$snowLayers                      $CLM_2010        ! (25) choice of method to combine and sub-divide snow layers (jrdn1991)
#!!!hc_profile                      constant        ! (12) choice of hydraulic conductivity profile
#!!!veg_traits                      CM_QJRMS1998    ! (17) choice of parameterization for vegetation roughness length and displacement height
#!!!canopyEmis                      difTrans        ! (18) choice of parameterization for canopy emissivity
#!!!snowIncept                      lightSnow       ! (19) choice of parameterization for snow interception
#!!!compaction                      anderson        ! (24) choice of compaction routine
! ***********************************************************************************************************************
All Models
! ***********************************************************************************************************************
#soilCatTbl                      ROSETTA         ! (03) soil-category dateset
#vegeParTbl                      USGS            ! (04) vegetation category dataset
#soilStress                      NoahType        ! (05) choice of function for the soil moisture control on stomatal resistance
#stomResist                      BallBerry       ! (06) choice of function for stomatal resistance
! ***********************************************************************************************************************
#num_method                      itertive        ! (07) choice of numerical method
#fDerivMeth                      analytic        ! (08) method used to calculate flux derivatives
#LAI_method                      monTable        ! (09) method used to determine LAI and SAI
#f_Richards                      mixdform        ! (10) form of Richard's equation
#groundwatr                      noXplict        ! (11) choice of groundwater parameterization
#!!!hc_profile                      constant        ! (12) choice of hydraulic conductivity profile
#bcUpprTdyn                      nrg_flux        ! (13) type of upper boundary condition for thermodynamics
#bcLowrTdyn                      zeroFlux        ! (14) type of lower boundary condition for thermodynamics
#bcUpprSoiH                      liq_flux        ! (15) type of upper boundary condition for soil hydrology
#bcLowrSoiH                      drainage        ! (16) type of lower boundary condition for soil hydrology
#!!!veg_traits                      CM_QJRMS1998    ! (17) choice of parameterization for vegetation roughness length and displacement height
#!!!canopyEmis                      difTrans        ! (18) choice of parameterization for canopy emissivity
#!!!snowIncept                      lightSnow       ! (19) choice of parameterization for snow interception
#windPrfile                      logBelowCanopy  ! (20) choice of wind profile through the canopy
#astability                      louisinv        ! (21) choice of stability function
#canopySrad                      CLM_2stream     ! (22) choice of canopy shortwave radiation method
#$$$alb_method                      varDecay        ! (23) choice of albedo representation
#!!!compaction                      anderson        ! (24) choice of compaction routine
#snowLayers                      CLM_2010        ! (25) choice of method to combine and sub-divide snow layers
#thCondSnow                      jrdn1991        ! (26) choice of thermal conductivity representation for snow
#thCondSoil                      mixConstit      ! (27) choice of thermal conductivity representation for soil
#spatial_gw                      localColumn     ! (28) choice of method for the spatial representation of groundwater
#subRouting                      timeDlay        ! (29) choice of method for sub-grid routing
#******************************************************************************************************************************
#'aliceblue':            '#F0F8FF',
#'antiquewhite':         '#FAEBD7',
#'aqua':                 '#00FFFF',
#'aquamarine':           '#7FFFD4',
#'azure':                '#F0FFFF',
#'beige':                '#F5F5DC',
#'bisque':               '#FFE4C4',
#'black':                '#000000',
#'blanchedalmond':       '#FFEBCD',
#'blue':                 '#0000FF',
#'blueviolet':           '#8A2BE2',
#'brown':                '#A52A2A',
#'burlywood':            '#DEB887',
#'cadetblue':            '#5F9EA0',
#'chartreuse':           '#7FFF00',
#'chocolate':            '#D2691E',
#'coral':                '#FF7F50',
#'cornflowerblue':       '#6495ED',
#'cornsilk':             '#FFF8DC',
#'crimson':              '#DC143C',
#'cyan':                 '#00FFFF',
#'darkblue':             '#00008B',
#'darkcyan':             '#008B8B',
#'darkgoldenrod':        '#B8860B',
#'darkgray':             '#A9A9A9',
#'darkgreen':            '#006400',
#'darkkhaki':            '#BDB76B',
#'darkmagenta':          '#8B008B',
#'darkolivegreen':       '#556B2F',
#'darkorange':           '#FF8C00',
#'darkorchid':           '#9932CC',
#'darkred':              '#8B0000',
#'darksalmon':           '#E9967A',
#'darkseagreen':         '#8FBC8F',
#'darkslateblue':        '#483D8B',
#'darkslategray':        '#2F4F4F',
#'darkturquoise':        '#00CED1',
#'darkviolet':           '#9400D3',
#'deeppink':             '#FF1493',
#'deepskyblue':          '#00BFFF',
#'dimgray':              '#696969',
#'dodgerblue':           '#1E90FF',
#'firebrick':            '#B22222',
#'floralwhite':          '#FFFAF0',
#'forestgreen':          '#228B22',
#'fuchsia':              '#FF00FF',
#'gainsboro':            '#DCDCDC',
#'ghostwhite':           '#F8F8FF',
#'gold':                 '#FFD700',
#'goldenrod':            '#DAA520',
#'gray':                 '#808080',
#'green':                '#008000',
#'greenyellow':          '#ADFF2F',
#'honeydew':             '#F0FFF0',
#'hotpink':              '#FF69B4',
#'indianred':            '#CD5C5C',
#'indigo':               '#4B0082',
#'ivory':                '#FFFFF0',
#'khaki':                '#F0E68C',
#'lavender':             '#E6E6FA',
#'lavenderblush':        '#FFF0F5',
#'lawngreen':            '#7CFC00',
#'lemonchiffon':         '#FFFACD',
#'lightblue':            '#ADD8E6',
#'lightcoral':           '#F08080',
#'lightcyan':            '#E0FFFF',
#'lightgoldenrodyellow': '#FAFAD2',
#'lightgreen':           '#90EE90',
#'lightgray':            '#D3D3D3',
#'lightpink':            '#FFB6C1',
#'lightsalmon':          '#FFA07A',
#'lightseagreen':        '#20B2AA',
#'lightskyblue':         '#87CEFA',
#'lightslategray':       '#778899',
#'lightsteelblue':       '#B0C4DE',
#'lightyellow':          '#FFFFE0',
#'lime':                 '#00FF00',
#'limegreen':            '#32CD32',
#'linen':                '#FAF0E6',
#'magenta':              '#FF00FF',
#'maroon':               '#800000',
#'mediumaquamarine':     '#66CDAA',
#'mediumblue':           '#0000CD',
#'mediumorchid':         '#BA55D3',
#'mediumpurple':         '#9370DB',
#'mediumseagreen':       '#3CB371',
#'mediumslateblue':      '#7B68EE',
#'mediumspringgreen':    '#00FA9A',
#'mediumturquoise':      '#48D1CC',
#'mediumvioletred':      '#C71585',
#'midnightblue':         '#191970',
#'mintcream':            '#F5FFFA',
#'mistyrose':            '#FFE4E1',
#'moccasin':             '#FFE4B5',
#'navajowhite':          '#FFDEAD',
#'navy':                 '#000080',
#'oldlace':              '#FDF5E6',
#'olive':                '#808000',
#'olivedrab':            '#6B8E23',
#'orange':               '#FFA500',
#'orangered':            '#FF4500',
#'orchid':               '#DA70D6',
#'palegoldenrod':        '#EEE8AA',
#'palegreen':            '#98FB98',
#'paleturquoise':        '#AFEEEE',
#'palevioletred':        '#DB7093',
#'papayawhip':           '#FFEFD5',
#'peachpuff':            '#FFDAB9',
#'peru':                 '#CD853F',
#'pink':                 '#FFC0CB',
#'plum':                 '#DDA0DD',
#'powderblue':           '#B0E0E6',
#'purple':               '#800080',
#'red':                  '#FF0000',
#'rosybrown':            '#BC8F8F',
#'royalblue':            '#4169E1',
#'saddlebrown':          '#8B4513',
#'salmon':               '#FA8072',
#'sandybrown':           '#FAA460',
#'seagreen':             '#2E8B57',
#'seashell':             '#FFF5EE',
#'sienna':               '#A0522D',
#'silver':               '#C0C0C0',
#'skyblue':              '#87CEEB',
#'slateblue':            '#6A5ACD',
#'slategray':            '#708090',
#'snow':                 '#FFFAFA',
#'springgreen':          '#00FF7F',
#'steelblue':            '#4682B4',
#'tan':                  '#D2B48C',
#'teal':                 '#008080',
#'thistle':              '#D8BFD8',
#'tomato':               '#FF6347',
#'turquoise':            '#40E0D0',
#'violet':               '#EE82EE',
#'wheat':                '#F5DEB3',
#'white':                '#FFFFFF',
#'whitesmoke':           '#F5F5F5',
#'yellow':               '#FFFF00',
#'yellowgreen':          '#9ACD32'}






































