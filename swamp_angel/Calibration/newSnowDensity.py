# -*- new snow density
#! compute density of new snow
 if(snowfall > tiny(fracrain))then
  #! Determine which method to use
  select case(model_decisions(iLookDECISIONS%snowDenNew)%iDecision)
   #! Hedstrom and Pomeroy 1998
   case(hedAndPom)
    newSnowDensity = min(pomNewSnowDenMax,newSnowDenMin + newSnowDenMult*exp((airtemp-Tfreeze)/newSnowDenScal))  ! new snow density (kg m-3)
   #! Pahaut 1976 (Boone et al. 2002)
   case(pahaut_76)
    newSnowDensity = max(newSnowDenMin,newSnowDenAdd + (newSnowDenMultTemp * (airtemp-Tfreeze))+(newSnowDenMultWind*((windspd)**pahautDenWindScal))); ! new snow density (kg m-3)
   #! Anderson 1976
   case(anderson)
    if(airtemp>(Tfreeze+andersonWarmDenLimit))then
     newSnowDensity = newSnowDenMin + newSnowDenMultAnd*(newSnowDenBase)**(andersonDenScal) ! new snow density (kg m-3)
    elseif(airtemp<=(Tfreeze-andersonColdDenLimit))then
     newSnowDensity = newSnowDenMin ! new snow density (kg m-3)
    else
     newSnowDensity = newSnowDenMin + newSnowDenMultAnd*(airtemp-Tfreeze+newSnowDenBase)**(andersonDenScal) ! new snow density (kg m-3)
    end if
   #! Constant new snow density
   case(constDens)
    newSnowDensity = constSnowDen ! new snow density (kg m-3)
   case default; message=trim(message)//'unable to identify option for new snow density'; err=20; return
  end select ! identifying option for new snow density
 else
  newSnowDensity = valueMissing
  rainfall = rainfall + snowfall ! in most cases snowfall will be zero here
  snowfall = 0._dp
 end if
 
 
 
 Tfreeze = 273.2 K
 pomNewSnowDenMax=150._dp   #! Upper limit for new snow density limit in Hedstrom and Pomeroy 1998. 150 was used because at was the highest observed density at air temperatures used in this study. See Figure 4 of Hedstrom and Pomeroy (1998).
 newSnowDensity = min(pomNewSnowDenMax,newSnowDenMin + newSnowDenMult*exp((airtemp-Tfreeze)/newSnowDenScal))  #! new snow density (kg m-3)
 
 newSnowDensity = constSnowDen ! new snow density (kg m-3)
 
 
 
 
 
 
 
 