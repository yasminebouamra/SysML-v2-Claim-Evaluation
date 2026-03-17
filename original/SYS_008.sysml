package Space_Launch_System{
 	doc
            /*
             * The Space Launch System consists of the SLS Rocket.
             */
    private import ScalarValues::*;
    private import SI::*;

package SLS_Rocket_Requirements {	
	requirement <'1'> Req1_SLS_Rocket {
		doc /* The Rocket shall provide the power to help Orion reach the speed needed to send it to the Moon. */
		requirement <'1.1'> Req1_1_Core_Stage {
			doc /* The Core Stage of the rocket shall facilitate the launch and achieve insertion into Earth orbit.*/
			<error>
			requirement <'1.1.1'> Req1_1_1_Core_Main_Propulsion {
				doc /* The Core stage Main Propulsion system shall supply propellants to the engines under the required temperature and pressure conditions.*/
				}
				</error>
			requirement <'1.1.2'> Req1_1_2_Core_Sections {
				doc /* The core stage shall be comprised of five major sections: the engine section, the liquid hydrogen tank, the intertank, the liquid oxygen tank, and the forward skirt.*/
				
				requirement <'1.1.2.1'> Req1_1_2_1_Engine_Section {
					doc /* The engine section shall house the engines, the engine thrust structure, and the propellant ducts.*/
						 
					requirement <'1.1.2.1.1'> Req1_1_2_1_1_Engine_Extremes {
						doc /* The engine shall be able to operate in the temoerature and pressure extremes the core stage will experience.*/
						}
					}
				requirement <'1.1.2.2'> Req1_1_2_2_Forward_Skirt_Components {
					doc /* The forward skirt shall house avionics systems, provide structural support, facilitate access and maintenance, manage internal pressure and venting, and ensure reliable communication.*/
					<error>
					requirement <'1.1.2.2.1'> Req1_1_2_2_1_Skirt_Additionals {
						doc /* The forward skirt shall have access doors, a ventilation system, pressurant lines, and communication antennas.*/
						}
					</error>
					requirement <'1.1.2.2.2'> Req1_1_2_2_2_Skirt_Avionics {
						doc /* The forward skirt shall house a majority of the rocket's avionics.*/
						}
					requirement <'1.1.2.2.3'> Req1_1_2_2_3_Skirt_Flight_System {
						doc /* The forward skirt shall house three redundant flight computers that serve as the flight control system for the rocket.*/
						}
					}
				requirement <'1.1.2.3'> Req1_1_2_3_Intertank {
					doc /* The intertank shall separate the upper dome of the liquid hydrogentank and the lower dome of the liquid oxygen tank and support the core stage structure.*/
					
					requirement <'1.1.2.3.1'> Req1_1_2_3_1_Intertank_Structure {
						doc /* The intertank shall contain a thrust structure to carry loads imparted from the solid rocket boosters during ascent.*/
						}
					requirement <'1.1.2.3.2'> Req1_1_2_3_2_Intertank_Interface {
						doc /* The intertank shall contain interfaces to the liquid hydrogen and liquid oxygen tanks.*/
						}
					}
				requirement <'1.1.2.4'> Req1_1_2_4_Liquid_Hydrogen {
					doc /* The liquid hydrogen tank shall have the capacity to fuel the engines from launch to Earth orbit insertion.*/
					}
				requirement <'1.1.2.5'> Req1_1_2_5_Liquid_Oxygen {
					doc /* The liquid oxygen tank shall have the capacity to fuel the engines from launch to Earth orbit insertion.*/
					}
				}
			<error>
			requirement <'1.1.3'> Req1_1_3_Core_Components {
				doc /* The core stage shall house the propellant tanks, engines, avionics, and attach points for a pair of solid rocket boosters necessary to launch from the ground and ascend to Earth orbit.*/
				}	
			</error>
			requirement <'1.1.4'> Req1_1_4_Rocket_Boosters {
				doc /* The SLS twin rocket boosters shall provide the necessary thrust and control during the initial phase of the launch.*/
					
				requirement <'1.1.4.1'> Req1_1_4_1_Rocket_Booster_Jettison {
					doc /* The SLS twin rocket boosters shall be jettisones after approximately two minutes of flight.*/
					}
				requirement <'1.1.4.2'> Req1_1_4_2_Rocket_Booster_Thrust {
					doc /* The SLS twin rocket boosters shall provide more than 75% of the total SLS thrust at launch and for the first two minutes of flight.*/
					}
				requirement <'1.1.4.3'> Req1_1_4_3_Rocket_Booster_Steering {
					doc /* The two solid-fuel, side-mounted booster motors shall provide the majority of steering for the rocket during the first two minutes of flight.*/
					}
				}
			requirement <'1.1.5'> Req1_1_5_Thermal_Protection_System {
				doc /* The thermal protection system shall provide adequate insulation to protect the core stage from temperature extremes and aerodynamic pressures during the mission.*/
				
				requirement <'1.1.5.1'> Req1_1_5_1_Aerodynamic_Pressures {
					doc /* The thermal protection system insulation shall be flexible enough to move with the rocket but rigid enough to take the aerodynamic pressures the SLS sustains as it launches into orbit.*/
					}
				requirement <'1.1.5.2'> Req1_1_5_2_Cryogenic_Insulation {
					doc /* The thermal protection system shall provide the insulation to keep the cryogenic propellant cold enough to remain liquid.*/
					}
				}
		}	
	<error>
	requirement <'1.2'> Req1_2_ICPS {
		doc /* The Interim Cryogenic Propulsuion Stage (ICPS) shall provide propulsion in space after the solid rocket boosters and core stage are jettisoned. */
		}
	</error>
	requirement <'1.3'> Req1_3_LVSA {
		doc /* The Launch Vehicle Stage Adapter (LVSA) shall connect the SLS core stage below to the Orion stage adapter above and support the operations of the rocket. */
		
		requirement <'1.3.1'> Req1_3_1_LVSA_ICPS {
		doc /* The LVSA shall protect avionics and electrical devices in the ICPS from vibration and acoustic conditions during launch and ascent.*/
		}
		requirement <'1.3.2'> Req1_3_2_LVSA_Separation {
		doc /* The LVSA shall provide a separation system such that once the core stage propellant is expended and the rocket is in Earth orbit, the core stage and LVSA is separated from the ICPS, Orion stage adapter, and Orion.*/
		}
		requirement <'1.3.3'> Req1_3_3_LVSA_Support {
		doc /* The LVSA shall provide structural support for launch.*/
		}
	}
	requirement <'1.4'> Req1_4_OSA {
		doc /* The Orion Stage Adapter (OSA) shall connect he ICPS to the Orion spacecraft and contain the secondary payload. */
		
		requirement <'1.4.1'> Req1_4_1_OSA_Secondary_Payload {
			doc /* The OSA shall carry small satellites, CubeSats, as seconary payloads.*/
			}
		requirement <'1.4.2'> Req1_4_2_OSA_Deployment_System {
			doc /* The OSA shall contain a secondary payload deployment system for CubeSats.*/
			
			requirement <'1.4.2.1'> Req1_4_2_1_OSA_Deployment {
			doc /* The OSA shall deplot CubeSats following separation from Orion and after the spacecraft is a safe distance away.*/
			}
		}
		<error>
		requirement <'1.4.3'> Req1_4_3_OSA_Disposal {
			doc /* The OSA shall remain attached to the ICPS as the ICPS continues on its disposal journey.*/
			}
		}
		</error>
	}
			
}


//Parts Package
package SpaceLaunchSystem_Parts {
	part SLSrocket{
		part coreStage{
			part propulsion;
			part engine;
			part forwardSkirt;
			part intertank;
			part oxygenTank;
			part hydrogenTank;			
		}
		part rocketBoosters;
		part thermalProtectionSystem;
		part ICPS;
		part LVSA;
		part OSA;
	}
}

//Requirements Satisfaction Package
	package satisfactionContext {
		private import SLS_Rocket_Requirements::*;
		private import SpaceLaunchSystem_Parts::*;
			
		satisfy Req1_SLS_Rocket by SLSrocket;
		satisfy Req1_SLS_Rocket.Req1_1_Core_Stage by SLSrocket.coreStage;
			satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_1_Core_Main_Propulsion by SLSrocket.coreStage.propulsion;
			satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_2_Core_Sections by SLSrocket.coreStage;
				satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_2_Core_Sections.Req1_1_2_1_Engine_Section by SLSrocket.coreStage.engine;
					satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_2_Core_Sections.Req1_1_2_1_Engine_Section.Req1_1_2_1_1_Engine_Extremes by SLSrocket.coreStage.engine;
				satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_2_Core_Sections.Req1_1_2_2_Forward_Skirt_Components by SLSrocket.coreStage.forwardSkirt;
					satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_2_Core_Sections.Req1_1_2_2_Forward_Skirt_Components.Req1_1_2_2_1_Skirt_Additionals by SLSrocket.coreStage.forwardSkirt;
					satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_2_Core_Sections.Req1_1_2_2_Forward_Skirt_Components.Req1_1_2_2_2_Skirt_Avionics by SLSrocket.coreStage.forwardSkirt;
					satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_2_Core_Sections.Req1_1_2_2_Forward_Skirt_Components.Req1_1_2_2_3_Skirt_Flight_System by SLSrocket.coreStage.forwardSkirt;
				satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_2_Core_Sections.Req1_1_2_3_Intertank by SLSrocket.coreStage.intertank;
					satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_2_Core_Sections.Req1_1_2_3_Intertank.Req1_1_2_3_1_Intertank_Structure by SLSrocket.coreStage.intertank;
					satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_2_Core_Sections.Req1_1_2_3_Intertank.Req1_1_2_3_2_Intertank_Interface by SLSrocket.coreStage.intertank;
				satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_2_Core_Sections.Req1_1_2_4_Liquid_Hydrogen by SLSrocket.coreStage.hydrogenTank;
				satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_2_Core_Sections.Req1_1_2_5_Liquid_Oxygen by SLSrocket.coreStage.oxygenTank;
			satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_3_Core_Components by SLSrocket.coreStage;
			satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_4_Rocket_Boosters by SLSrocket.rocketBoosters;
				satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_4_Rocket_Boosters.Req1_1_4_1_Rocket_Booster_Jettison by SLSrocket.rocketBoosters;
				satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_4_Rocket_Boosters.Req1_1_4_2_Rocket_Booster_Thrust by SLSrocket.rocketBoosters;
				satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_4_Rocket_Boosters.Req1_1_4_3_Rocket_Booster_Steering by SLSrocket.rocketBoosters;
			satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_5_Thermal_Protection_System by SLSrocket.thermalProtectionSystem;
				satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_5_Thermal_Protection_System.Req1_1_5_1_Aerodynamic_Pressures by SLSrocket.thermalProtectionSystem;
				satisfy Req1_SLS_Rocket.Req1_1_Core_Stage.Req1_1_5_Thermal_Protection_System.Req1_1_5_2_Cryogenic_Insulation by SLSrocket.thermalProtectionSystem;
		satisfy Req1_SLS_Rocket.Req1_2_ICPS by SLSrocket.ICPS;
		satisfy Req1_SLS_Rocket.Req1_3_LVSA by SLSrocket.LVSA;
			satisfy Req1_SLS_Rocket.Req1_3_LVSA.Req1_3_1_LVSA_ICPS by SLSrocket.LVSA;
			satisfy Req1_SLS_Rocket.Req1_3_LVSA.Req1_3_2_LVSA_Separation by SLSrocket.LVSA;
			satisfy Req1_SLS_Rocket.Req1_3_LVSA.Req1_3_3_LVSA_Support by SLSrocket.LVSA;		
		satisfy Req1_SLS_Rocket.Req1_4_OSA by SLSrocket.OSA; 
			satisfy Req1_SLS_Rocket.Req1_4_OSA.Req1_4_1_OSA_Secondary_Payload by SLSrocket.OSA; 
			satisfy Req1_SLS_Rocket.Req1_4_OSA.Req1_4_2_OSA_Deployment_System by SLSrocket.OSA; 
				satisfy Req1_SLS_Rocket.Req1_4_OSA.Req1_4_2_OSA_Deployment_System.Req1_4_2_1_OSA_Deployment by SLSrocket.OSA; 
			satisfy Req1_SLS_Rocket.Req1_4_OSA.Req1_4_3_OSA_Disposal by SLSrocket.OSA; 
	}
}

