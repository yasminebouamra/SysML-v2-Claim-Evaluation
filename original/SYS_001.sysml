package SatelliteRequirements {
	private import RequirementDerivation::*;
	private import NumericalFunctions::sum;
	
	part mission {
		attribute duration;
		attribute cost;
		attribute IOC;
	}
	
	part def Satellite {
		attribute mass :> SI::kg;
	}
	
	part sat : Satellite {
		attribute :>> mass = sum(subsystemmasses);
		attribute subsystemmasses[*] :> SI::kg;
		
		<error>part Payload {
			attribute mass :> subsystemmasses = 55 [SI::kg];
			attribute resolution = 0.5 [SI::m];
			attribute sw = 8 [SI::km];
		}
		</error>
		
		part ADCS {
			attribute mass :> subsystemmasses = 75 [SI::kg];
		}
		
		part Comm {
			attribute mass :> subsystemmasses = 50 [SI::kg];
		}
		
		part CDH {
			attribute mass :> subsystemmasses = 70 [SI::kg];
		}
		
		part Power {
			attribute mass :> subsystemmasses = 85 [SI::kg];
		}
		
		part Thermal {
			attribute mass :> subsystemmasses = 35 [SI::kg];
		}
		
		part Structures {
			attribute mass :> subsystemmasses = 150 [SI::kg];
		}
		<error>
		part Propulsion {
			attribute mass :> subsystemmasses = 55 [SI::kg];
		}
		</error>
	}
	
	requirement <'1'> Functional {
		doc /*Functional Requirements for the Satellite */
		require satMassReqt;
		require ResolutionReqt;
		require SWReqt;
	}
	
	requirement <'2'> Operational {
		doc /*Operational Requirements for the Satellite */
		require DurationReqt;
	}
	
	requirement <'3'> Constraints {
		doc /*Contraints for the Satellite */
		require CostReqt;
		require ScheduleReqt;
	}
	<error>
	requirement def MassReqt {
		doc /* The actual mass shall be less than or equal to the required mass. */
		subject mass :> SI::kg;
		attribute massLimit :> SI::kg;
		require constraint { mass <= massLimit }
	}
	</error>
	requirement <'1.1'> satMassReqt : MassReqt {
		subject mass :> SI::kg;
		attribute :>> massLimit = 600[SI::kg];
	}
	
	requirement <'1.1.1'> ADCSMassReqt : MassReqt {
		subject :>> mass = sat.ADCS.mass;
	}
	
	requirement <'1.1.2'> CDHMassReqt : MassReqt {
		subject :>> mass = sat.CDH.mass;
	}
	
	requirement <'1.1.3'> CommMassReqt : MassReqt {
		subject :>> mass = sat.Comm.mass;
	}
	
	requirement <'1.1.4'> PayloadMassReqt : MassReqt {
		subject :>> mass = sat.Payload.mass;
	}
	
	requirement <'1.1.5'> PowerMassReqt : MassReqt {
		subject :>> mass = sat.Power.mass;
	}
	
	requirement <'1.1.6'> PropulsionMassReqt : MassReqt {
		subject :>> mass = sat.Propulsion.mass;
	}
	
	requirement <'1.1.7'> StructuresMassReqt : MassReqt {
		subject :>> mass = sat.Structures.mass;
	}                                                                           

	requirement <'1.1.8'> ThermalMassReqt : MassReqt {
		subject :>> mass = sat.Thermal.mass;
	}
	
	#derivation connection {
		end #original ::> satMassReqt;
		end #derive ::> PayloadMassReqt;
		end #derive ::> ADCSMassReqt;
		end #derive ::> CDHMassReqt;
		end #derive ::> PropulsionMassReqt;
		end #derive ::> PowerMassReqt;
		end #derive ::> StructuresMassReqt;
		end #derive ::> CommMassReqt;
		end #derive ::> ThermalMassReqt;
	}
	
	requirement <'1.2'> ResolutionReqt {
		doc /* The camera shall have a resolution of 1 meter or less. */
		subject resolution :> SI::m;
		require constraint { resolution <= 1[SI::m] }
	}
	
	requirement <'1.3'> SWReqt {
		doc /* The camera shall have a swath width of at least 7.5 kilometers. */
		subject sw :> SI::km;
		require constraint { sw >= 4[SI::km] }
	}
	<error>
	requirement <'2.1'> DurationReqt {
		doc /* The satellite shall operate for at least 10 years. */
		subject duration;
		require constraint { duration >= 10 }
	}
	</error>
	requirement <'3.1'> CostReqt {
		doc/* The cost for R+D shall be less than $20M. */
		subject cost;
		require constraint { cost <= 20 }
	}
	
	requirement <'3.2'> ScheduleReqt {
		doc /* The satellite shall have initial operating capabilities within 1.5 years, with final capabilities within 2 years. */
		subject IOC;
		require constraint { IOC <= 1.5 }
	}
}