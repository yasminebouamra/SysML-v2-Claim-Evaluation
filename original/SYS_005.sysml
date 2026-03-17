package Orion_Spacecraft{
 	doc
            /*
             * The Orion Spacecraft consists of the Orion module to carry astronauts on deep space missions.
             */
    private import ScalarValues::*;
    private import SI::*;

package Orion_Requirements {
    requirement <'2'> Req2_Orion {
        doc /* The Orion spacecraft shall be capable of safely carrying astronauts on deep space exploration missions. */

        requirement <'2.1'> Req2_1_Crew_Module {
            doc /* The Orion spacecraft shall have a crew module designed for astronauts to live and work in. */

            requirement <'2.1.1'> Req2_1_1_Supplies {
                doc /* The crew module shall be able to accommodate four crew members for up to 21 days without docking with another spacecraft. */
            }
            requirement <'2.1.2'> Req2_1_2_Crew_Propulsion {
                doc /* The crew module shall have a propulsion system capable of providing full control of crew module translation and rotation. */
            }
            requirement <'2.1.3'> Req2_1_3_Crew_Safety {
                doc /* The crew module shall provide a safe habitat for crew members through launch, in-orbit operations, landing, and recovery. */
            }
            requirement <'2.1.4'> Req2_1_4_Crew_Parachute {
                doc /* The top of the crew module shall have a parachute system to be deployed after re-entering Earth's atmosphere that slows down the module to a safe speed for splashdown. */
            }
        }
        <error>
        requirement <'2.2'> Req2_2_Service_Module {
            doc /* The Orion spacecraft shall have a service module that provides power, propulsion, and critical supplies. */

            requirement <'2.2.1'> Req2_2_1_Consumable_Storage {
                doc /* The consumable storage system of the service module shall provide potable water, nitrogen, and oxygen to the crew module. */
            }
        </error>
            requirement <'2.2.2'> Req2_2_2_Solar_Array_Deployment {
                doc /* The service module shall unfurl its solar arrays once Orion is above the atmosphere. */

                requirement <'2.2.2.1'> Req2_2_2_1_Solar_Array_Battery {
                    doc /* The solar arrays shall charge batteries in the crew module. */

                    requirement <'2.2.2.1.1'> Req2_2_2_1_1_Battery_Power {
                        doc /* The Orion spacecraft shall operate off of battery power if the solar arrays cannot be depended upon. */
                    }
                }
            }
            requirement <'2.2.3'> Req2_2_3_Service_Propulsion {
                doc /* The service module shall propel Orion after the spacecraft separates from the upper stage of the SLS rocket. */

                requirement <'2.2.3.1'> Req2_2_3_1_Service_Navigation {
                    doc /* The service module shall maintain the course of Orion as it travels to the Moon and back. */
                }
            }
            requirement <'2.2.4'> Req2_2_4_Service_Detachment {
                doc /* The service module shall detach from the crew module before it enters Earth's atmosphere. */
            }
            requirement <'2.2.5'> Req2_2_5_Service_TCS {
                doc /* The service module's thermal control system shall keep the equipment and crew members at a comfortable temperature. */
            }
        }
        <error>
        requirement <'2.3'> Req2_3_LAS {
            doc /* The Orion spacecraft shall have a launch abort system that can pull the spacecraft and crew to safety in the event of an emergency during launch or ascent to orbit. */
        </error>
            requirement <'2.3.1'> Req2_3_1_LAS_Actuation {
                doc /* The LAS shall pull the spacecraft away from the rocket and position the module for a safe landing. */

                requirement <'2.3.1.1'> Req2_3_1_1_Abort_Motor {
                    doc /* The abort motor shall be able to pull the crew module away from the rocket. */
                }
                requirement <'2.3.1.2'> Req2_3_1_2_Attitude_Control_Motor {
                    doc /* The attitude control motor shall be able to steer and orient the capsule precisely. */
                }
                requirement <'2.3.1.3'> Req2_3_1_3_Jettison_Motor {
                    doc /* The jettison motor shall be able to ignite to separate the LAS from the crew module prior to parachute deployment. */
                }
            }
            requirement <'2.3.2'> Req2_3_2_LAS_Activation {
                doc /* The launch abort system (LAS) shall be able to activate within milliseconds. */
            }
        }

        requirement <'2.4'> Req2_4_Orion_Communication_Nav {
            doc /* The Orion spacecraft shall be capable of communicating and navigating beyond the range of GPS and above communication satellites in Earth orbit. */
        }

        requirement <'2.5'> Req2_5_Orion_Radiation_Shielding {
            doc /* The Orion spacecraft shall have sufficient radiation shielding to protect its internals. */
        }
        <error>
        requirement <'2.6'> Req2_6_Orion_Heat_Shielding {
            doc /* The Orion spacecraft shall have sufficient heat shielding to protect its internals during high-speed re-entry into Earth's atmosphere. */
        </error>
            requirement <'2.6.1'> Req2_6_1_Reentry_Temperatures {
                doc /* The Orion spacecraft shall have sufficient ablative heat shielding to endure temperatures of nearly 5,000 degrees Fahrenheit. */
            }
        }

        requirement <'2.7'> Req2_7_Orion_Flight_Software {
            doc /* The Orion primary flight computer software shall be fully automated such that it can execute the mission and handle potential faults. */
        }
    }
}


//Parts Package
package Orion_Parts {
    part Orion {
        part crewModule {
            attribute suppliesNeeded :> ISQ::time;
            part crewPropulsion;
            part crewSafety;
            part crewParachute;
        }

        part serviceModule {
            part consumableStorage;

            part solarArrayDeployment {
                part solarArrayBattery {
                    part batteryPower;
                }
            }

            part servicePropulsion {
                part serviceNavigation;
            }

            part serviceDetachment;
            part serviceTCS{
            	attribute serviceTCSTemp :> ISQ::temperature;
            }
        }

        part LAS {
            part LASActuation {
                part abortMotor;
                part attitudeControlMotor;
                part jettisonMotor;
            }
            attribute LASActivationTime :> ISQ::time;
        }

        part orionCommunicationNav;
        part orionRadiationShielding;

        part orionHeatShielding {
            attribute temperature :> ISQ::temperature;
        }

        part orionFlightSoftware;
    }
}

//Requirements Satisfaction Package
package satisfactionContext {
    private import Orion_Requirements::*;
    private import Orion_Parts::*;

    satisfy Req2_Orion by Orion;
        satisfy Req2_Orion.Req2_1_Crew_Module by Orion.crewModule;
            satisfy Req2_Orion.Req2_1_Crew_Module.Req2_1_1_Supplies by Orion.crewModule;
            satisfy Req2_Orion.Req2_1_Crew_Module.Req2_1_2_Crew_Propulsion by Orion.crewModule.crewPropulsion;
            satisfy Req2_Orion.Req2_1_Crew_Module.Req2_1_3_Crew_Safety by Orion.crewModule.crewSafety;
            satisfy Req2_Orion.Req2_1_Crew_Module.Req2_1_4_Crew_Parachute by Orion.crewModule.crewParachute;

        satisfy Req2_Orion.Req2_2_Service_Module by Orion.serviceModule;
            satisfy Req2_Orion.Req2_2_Service_Module.Req2_2_1_Consumable_Storage by Orion.serviceModule.consumableStorage;
            satisfy Req2_Orion.Req2_2_Service_Module.Req2_2_2_Solar_Array_Deployment by Orion.serviceModule.solarArrayDeployment;
                satisfy Req2_Orion.Req2_2_Service_Module.Req2_2_2_Solar_Array_Deployment.Req2_2_2_1_Solar_Array_Battery by Orion.serviceModule.solarArrayDeployment.solarArrayBattery;
                    satisfy Req2_Orion.Req2_2_Service_Module.Req2_2_2_Solar_Array_Deployment.Req2_2_2_1_Solar_Array_Battery.Req2_2_2_1_1_Battery_Power by Orion.serviceModule.solarArrayDeployment.solarArrayBattery.batteryPower;
            satisfy Req2_Orion.Req2_2_Service_Module.Req2_2_3_Service_Propulsion by Orion.serviceModule.servicePropulsion;
                satisfy Req2_Orion.Req2_2_Service_Module.Req2_2_3_Service_Propulsion.Req2_2_3_1_Service_Navigation by Orion.serviceModule.servicePropulsion.serviceNavigation;
            satisfy Req2_Orion.Req2_2_Service_Module.Req2_2_4_Service_Detachment by Orion.serviceModule.serviceDetachment;
            satisfy Req2_Orion.Req2_2_Service_Module.Req2_2_5_Service_TCS by Orion.serviceModule.serviceTCS;

        satisfy Req2_Orion.Req2_3_LAS by Orion.LAS;
            satisfy Req2_Orion.Req2_3_LAS.Req2_3_1_LAS_Actuation by Orion.LAS.LASActuation;
                satisfy Req2_Orion.Req2_3_LAS.Req2_3_1_LAS_Actuation.Req2_3_1_1_Abort_Motor by Orion.LAS.LASActuation.abortMotor;
                satisfy Req2_Orion.Req2_3_LAS.Req2_3_1_LAS_Actuation.Req2_3_1_2_Attitude_Control_Motor by Orion.LAS.LASActuation.attitudeControlMotor;
                satisfy Req2_Orion.Req2_3_LAS.Req2_3_1_LAS_Actuation.Req2_3_1_3_Jettison_Motor by Orion.LAS.LASActuation.jettisonMotor;
            satisfy Req2_Orion.Req2_3_LAS.Req2_3_2_LAS_Activation by Orion.LAS;

        satisfy Req2_Orion.Req2_4_Orion_Communication_Nav by Orion.orionCommunicationNav;
        satisfy Req2_Orion.Req2_5_Orion_Radiation_Shielding by Orion.orionRadiationShielding;

        satisfy Req2_Orion.Req2_6_Orion_Heat_Shielding by Orion.orionHeatShielding;
            satisfy Req2_Orion.Req2_6_Orion_Heat_Shielding.Req2_6_1_Reentry_Temperatures by Orion.orionHeatShielding;

        satisfy Req2_Orion.Req2_7_Orion_Flight_Software by Orion.orionFlightSoftware;
}

}

