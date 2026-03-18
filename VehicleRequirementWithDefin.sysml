package VehicleRequirementWithDefinitions {
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
        require constraint {
            mass <= massLimit
        }
    }
    requirement <'MAS-001'> 'Vehicle Mass Requirement' : MassRequirement {
        doc /* The vehicle shall have a maximum mass of at most 2200 kg.*/
        subject :>> mass = vehicle.mass;
        attribute :>> massLimit = 2200 [kg];
    }
    requirement <'MAS-002'> 'Chassis Mass Requirement' : MassRequirement {
        doc /* The vehicle chassis shall have a maximum mass of at most 1200 kg.*/
        subject :>> mass = vehicle.chassis.mass;
        attribute :>> massLimit = 1200 [kg];
    }
    requirement <'MAS-003'> 'Engine Mass Requirement' : MassRequirement {
        doc /* The vehicle engine shall have a maximum mass of at most 500 kg.*/
        subject :>> mass = vehicle.engine.mass;
        attribute :>> massLimit = 500 [kg];
    }
    requirement def BatteryAutonomyRequirement {
        subject autonomy :> ISQ::length;
        attribute autonomyMin :> ISQ::length;
        attribute autonomyMax :> ISQ::length;
        attribute temperature :> ISQ::temperature;
        assume constraint {
            temperature >= 20 ['°C']
        }
        require constraint {
            autonomy >= autonomyMin
        }
        require constraint {
            autonomy <= autonomyMax
        }
    }
    requirement <'AUT-001'> 'Autonomy Requirement Ambient Temperature' : BatteryAutonomyRequirement {
        doc /* The vehicle shall provide an autonomy between 600 km and 800 km under ambient temperature of 20 °C.*/
        subject :>> autonomy = vehicle.battery.autonomy;
        attribute :>> autonomyMin = 600 [km];
        attribute :>> autonomyMax = 800 [km];
        attribute :>> temperature = 20 ['°C'];
    }
    requirement <'AUT-002'> 'Autonomy Requirement Cold Temperature' : BatteryAutonomyRequirement {
        doc /* The vehicle shall provide an autonomy between 400 km and 500 km under ambient temperature of 4 °C.*/
        subject :>> autonomy = vehicle.battery.autonomy;
        attribute :>> autonomyMin = 400 [km];
        attribute :>> autonomyMax = 500 [km];
        attribute :>> temperature = 4 ['°C'];
    }
}
