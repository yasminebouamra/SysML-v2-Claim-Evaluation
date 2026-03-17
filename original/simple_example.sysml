package VehicleRequirementDerivation {
    private import ISQ::**;
    public import SI::**;
    part vehicle {
        attribute mass :> ISQ::mass; 
        part chassis {
            attribute mass :> ISQ::mass; 
        }

        part engine {
            attribute mass :> ISQ::mass; 
        }

        part battery {
            attribute autonomy :> ISQ::length; 
        }
    }

    requirement def MassRequirement {
        subject mass :> ISQ::mass; 
        attribute massLimit :> ISQ::mass;
        require constraint { mass <= massLimit }
    }

    <error>
    requirement vehicleMassRequirement : MassRequirement {
        subject :>> mass = vehicle.mass;
        attribute massLimit = 2000[kg];
    }
    </error>

    requirement chassisMassRequirement : MassRequirement {
        subject :>> mass = vehicle.chassis.mass;
        attribute massLimit = 1200[kg]; 
    }

    requirement engineMassRequirement : MassRequirement {
        subject :>> mass = vehicle.engine.mass;
        massLimit = 500[kg]; 
    }

    requirement def BatteryAutonomyRequirement {
        subject autonomy :> ISQ::length; 
        attribute autonomyMin :> ISQ::length;
        attribute autonomyMax :> ISQ::length;
        attribute temperature :> ISQ::temperature;
        
        require constraint {
            autonomy >= autonomyMin }
        require constraint {
            autonomy <= autonomyMax }
    }

    requirement temp20AutonomyRequirement : BatteryAutonomyRequirement {
        subject :>> autonomy = vehicle.battery.autonomy;
        attribute autonomyMin = 600[km];
        attribute autonomyMax = 800[km];
        attribute temperature = 20[C];
    }


    requirement temp4AutonomyRequirement : BatteryAutonomyRequirement {
        subject :>> autonomy = vehicle.battery.autonomy;
        attribute autonomyMin = 400[km];
        attribute autonomyMax = 500[km];
        attribute temperature = 4[C];
    }

}