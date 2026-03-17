package UAVDefinitions {
	doc
	/*
	 * Organ Transplant UAV definitions model.
	 */

	private import ScalarValues::*;
	private import Quantities::*;
	private import MeasurementReferences::*;
	private import ISQ::*;
	private import SI::*;
	
	/* REQUIREMENTS DEFINITIONS	 */
	
	requirement def <'1'> Mission {
		doc /*The OT-UAV shall transport the Organ Carrier from a donor hospital to the recipient hospital.*/
		}
		requirement def <'1.1'> Endurance {
			doc /*The OT-UAV shall have a maxium endurance of XX hours on a fully charged battery.*/
			}
		requirement def <'1.2'> Transport {
			doc /*The OT-UAV shall house the Organ Carrier, battery, and communications systems.*/
			}
		requirement def <'1.3'> Speed_of_Transport {
			doc /*The OT-UAV shall transport the organ within the viable time window for the type of organ being transported.*/
			}
	requirement def <'2'> Autonomous_Operation {
		doc /*The OT-UAV shall have fully autonomous take-off, flight, and landing capabilities.*/
		}
	<error>
	requirement def <'3'> Human_Override {
		doc /*The OT-UAV shall have a human-controller override.*/
		}
	</error>
	requirement def <'4'> Organ_Carrier {
		doc /*The Organ Carier shall house and auxiliary battery, Organ Monitoring System, and the Organ.*/
		}
		requirement def <'4.1'> Organ_Carrier_Structure {
			doc /*The Organ Carrier shall be structurally resilient to OT-UAV ground and inflight temperatures, forces, and vibrations.*/
			}
		requirement def <'4.2'> Organ_Carrier_Organ_Requirements {
			doc /*The Organ Carrier shall house the Organ in organ-specific requirements for orientation, temperature, forces, and vibrations.*/
			}
		requirement def <'4.3'> Organ_Monitoring_System_Reporting {
			doc /*The Organ Monitoring System shall monitor and report the temporal data for temperature, forces, vibrations, and location of the Organ Carrier.*/
			}
			requirement def <'4.3.1'> Organ_Monitoring_System_Comm_Channels {
				doc /*The Organ Monitoring System communications channel shall not interfere with OT-UAV operations communications channel.*/
				}
	requirement def <'5'> Rescue_System_Engage_during_Malfunction {
		doc /*The Organ Carrier shall be equipped with an Rescue System that will protect the Organ and safely land on the ground in case of a midair malfunction.*/
		}
		requirement def <'5.1'> Rescue_System_Protection {
			doc /*The Rescue System shall deliver the Organ Carrier gently to the ground to protect the Organ.*/
			}
		requirement def <'5.2'> Carrier_Mount_Disengage_during_Malfunction {
			doc /*The Carrier Mount shall disengage the Organ Carrier in case of a midair malfunction.*/
			}
		requirement def <'5.3'> Organ_Monitoring_System_Operate_during_Malfunction {
			doc /*The Organ Monitoring System shall monitor and report the temporal temperature, forces, vibrations, and location of the Organ Carrier in case of a midair malfunction.*/
			}
	<error>		
	requirement def <'6'> Carrier_Mount {
		doc /*The OT-UAV shall have a Carrier Mount as an interface to the Organ Carrier.*/
		}
	</error>
		requirement def <'6.1'> Carrier_Mount_Organ_Carrier {
			doc /*The Carrier Mount shall be structurally resilient to house the Organ Carrier.*/
			}
		requirement def <'6.2'> Carrier_Mount_Operations {
			doc /*The Carrier Mount shall be structurally resilient to temperature, forces, and vibrations of the OT-UAV during ground and inflight operations.*/
			}
	requirement def <'7'> Security_of_Organ_Carrier {
		doc /*Access to the Organ Carrier shall only be granted to eligible individuals to prevent theft.*/
		}
	requirement def <'8'> Weather_Operations {
		doc /*The OT-UAV shall be resilient to operate in a wide range of flight-safe weather conditions.*/
		}
	requirement def <'9'> Day_Night_Operations {
		doc /*The OT-UAV shall be operable during the day and the night.*/
		}
	requirement def <'10'> Beyond_LOS {
		doc /*The OT-UAV shall be operable beyond line-of-sight (BLOS).*/
		}
	requirement def <'11'> Urban_Air_Regulations {
		doc /*The OT-UAV shall comply will all FAA regulations.*/
		}
	requirement def <'12'> MassLimitationRequirement {
		doc /* The actual mass shall be less than or equal to the required mass. */
		
		attribute massActual: MassValue;
		attribute massReqd: MassValue;
		
		require constraint {massActual <= massReqd}
	}
	
	
	/* PART DEFINITIONS */
	
	part def UAV {
		attribute mass = airframe.mass + payload.mass + propulsionSystem.mass + flightComputer.mass + powerSystem.mass + communicationSystem.mass;
		
		part airframe: Airframe;
		part propulsionSystem: PropulsionSystem;
		part flightComputer: FlightControlSystem;
		part payload: Payload;
		part powerSystem: PowerSystem;
		part communicationSystem: CommunicationSystem;
		part groundComputer: GroundControlSystem;
	}
	
	part def Airframe{
		// Provides structural integrity, aerodynamic surfaces, mounts payload
		attribute mass:> ISQ::mass;
		
		part fuselage: Fuselage;
		part wings: Wings;	
		part empennage: Empennage;
		part landingGear: LandingGear;
		part mount: PayloadMount;
	}
	<error>
	part def Fuselage;
	part def Wings;
	part def Empennage;
	part def LandingGear;
	part def PayloadMount;
	</error>
	part def PropulsionSystem{
		attribute mass:> ISQ::mass;
		
		part motor: Motor;
		part propeller: Propeller;
		part pcs: PropulsionControlSystem;
	}
	
	part def Motor{
		//port power: MotorIF;
		//port torque: MotorIF;
	}
	part def Propeller{
		//port torque: PropellerIF;
		//port thrust: PropellerIF;
	}
	part def PropulsionControlSystem{
		//port powerInPort:PropulsionControlSystemIF;
		//port powerOutPort:PropulsionControlSystemIF;
	}
	
	part def FlightControlSystem{
		attribute mass:> ISQ::mass;
		
		part autopilot: FlightController;
		part imu: InertialMeasurementUnit;
		part servos: Actuators;
		part gpsRx: GPSReceiver;
		part magSensor: Magnetometer;
	}
	
	part def FlightController;
	part def InertialMeasurementUnit;
	part def Actuators;
	part def GPSReceiver;
	part def Magnetometer;
	
	
	part def Payload{
		attribute mass:> ISQ::mass;
		
		part organ: Organ;
		part carrier: OrganCarrier;
		part carrierMount: CarrierMount;
	}
	
	part def Organ;
	part def OrganCarrier;
	part def CarrierMount;
	
	part def GroundControlSystem{
		part operator: OperatorController;
		part groundAntenna: GroundAntenna;
		part telemetry: TelemetrySystem;
	}
	
	part def OperatorController;
	part def GroundAntenna;
	part def TelemetrySystem;
	
	part def PowerSystem{
		attribute mass:> ISQ::mass;
		
		part battery: PowerSource;
		part pdu: PowerDistributionBoard;
	}
	
	part def PowerSource;
	part def PowerDistributionBoard{
		//port powerInPort:PowerDistributionBoardIF;
		//port propPowerOutPort:PowerDistributionBoardIF;
		//port flightComputerPowerOutPort : PowerDistributionBoardIF;
	}

	<error>
	part def CommunicationSystem{
		attribute mass:> ISQ::mass;
		
		part uavAntenna: TxRxAntenna;
		part rfModule: RFModule;
	}
	</error>
	
	part def TxRxAntenna;
	part def RFModule;

}


