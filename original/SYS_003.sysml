package VehicleRequirementDerivation {
	private import RequirementDerivation::*;
	
	part vehicle {
		attribute mass :> ISQ::mass;
		
		part chassis {
			attribute mass :> ISQ::mass;
		}
		
		part engine {
			attribute mass :> ISQ::mass;
		}
	}
	
	requirement def MassRequirement {
		subject mass :> ISQ::mass;
		attribute massLimit :> ISQ::mass;
		require constraint { mass <= massLimit }
	}
	
	requirement vehicleMassRequirement : MassRequirement {
		subject :>> mass = vehicle.mass;
	}
	
	requirement chassisMassRequirement : MassRequirement {
		subject :>> mass = vehicle.chassis.mass;
	}
	
	requirement engineMassRequirement : MassRequirement {
		subject :>> mass = vehicle.engine.mass;
	}
	
	#derivation connection {
		end #original ::> vehicleMassRequirement;
		end #derive ::> chassisMassRequirement;
		end #derive ::> engineMassRequirement;
	}
	
}

package 'Saturn V Requirements' {

  public import ScalarValues::*;
  public import ISQ::*;
  public import SI::*;
  public import Requirements::*;
  public import Base::*;
  public import RequirementDerivation::*;
  <error>
  part saturnV {
    attribute totalMass :> ISQ::mass;
    attribute height :> ISQ::length;
    
    part stage1 {
      attribute thrust :> ISQ::force;
    }
    </error>
    part stage2 {
      attribute thrust :> ISQ::force;
    }
    part stage3 {
      attribute thrust :> ISQ::force;
      attribute ignitions : Integer;
    }
    
    part payload {
      attribute massLEO :> ISQ::mass;
      attribute massTLI :> ISQ::mass;
    }
  }

  requirement def MassRequirement {
    doc /* Ensures the mass of the subject does not exceed the massLimit. */
    subject m :> ISQ::mass;
    attribute massLimit : MassValue;
    require constraint { m <= massLimit }
  }

  requirement def ThrustRequirement {
    doc /* Ensures that the applied thrust meets or exceeds a required minimum value. */
    subject f :> ISQ::force;
    attribute thrustMin : ForceValue;
    require constraint { f >= thrustMin }
  }

  requirement def HeightRequirement {
    doc /* Verifies that the structure height matches the expected mission design height. */
    subject h :> ISQ::length;
    attribute expectedHeight : LengthValue;
    require constraint { h == expectedHeight }
  }
  <error>
  requirement def IgnitionRequirement {
    doc /* Specifies the minimum number of ignition cycles required for restart capability. */
    subject ign : Integer;
    attribute minIgnitions : Integer = 2;
    require constraint { ign >= minIgnitions }
  }
  </error>

  requirement totalMassRequirement : MassRequirement {
    doc /* Limits the total vehicle mass to ensure lift-off capability. */
    subject :>> m = saturnV.totalMass;
    massLimit = 2970000[kg];
  }

  requirement heightRequirement : HeightRequirement {
    doc /* Sets the required Saturn V height for structural and aerodynamic stability. */
    subject :>> h = saturnV.height;
    expectedHeight = 110.0[m];
  }
  <error>
  requirement stage1ThrustRequirement : ThrustRequirement {
    doc /* Ensures that Stage I engines provide sufficient initial launch thrust. */
    subject :>> f = saturnV.stage1.thrust;
    thrustMin = 34e6[N];
  }
  </error>

  requirement stage2ThrustRequirement : ThrustRequirement {
    doc /* Ensures that Stage II engines provide required mid-ascent acceleration. */
    subject :>> f = saturnV.stage2.thrust;
    thrustMin = 5e6[N];
  }

  requirement stage3ThrustRequirement : ThrustRequirement {
    doc /* Ensures that Stage III engines are powerful enough for orbit insertion. */
    subject :>> f = saturnV.stage3.thrust;
    thrustMin = 1e6[N];
  }

  requirement stage3IgnitionRequirement : IgnitionRequirement {
    doc /* Confirms that Stage III supports multiple ignitions for TLI maneuvers. */
    subject :>> ign = saturnV.stage3.ignitions;
  }

  requirement payloadLEORequirement : MassRequirement {
    doc /* Limits the mass of the Low Earth Orbit payload. */
    subject :>> m = saturnV.payload.massLEO;
    massLimit = 118000[kg];
  }

  <error>
  requirement payloadTLIRequirement : MassRequirement {
    doc /* Limits the mass of the payload inserted into Trans-Lunar Injection (TLI). */
    subject :>> m = saturnV.payload.massTLI;
    massLimit = 43500[kg];
  }
  </error>
  requirement launchCapability {
    doc /* Bundles all primary vehicle performance requirements needed for a successful launch. */
    requirement totalMassRequirement;
    requirement heightRequirement;
    requirement stage1ThrustRequirement;
    requirement stage2ThrustRequirement;
    requirement stage3ThrustRequirement;
    requirement stage3IgnitionRequirement;
    requirement payloadLEORequirement;
    requirement payloadTLIRequirement;
  }

  #derivation connection massDerivation {
    end #original ::> totalMassRequirement;
    end #derive ::> payloadLEORequirement;
    end #derive ::> payloadTLIRequirement;
  }

}

