package Propeller_Aircraft{
 	doc
            /*
             * The propeller aircraft model consists of aircraft requirements and satisfaction contexts wiwth various parts.
             */
    private import ScalarValues::*;
    private import SI::*;

package Aircraft_Requirements {	
	requirement <'1'> Req1_Range {
        doc /* The aircraft shall have a maximum range of at least 1000 km. */
    }

    requirement <'2'> Req2_Endurance {
        doc /* The aircraft shall have a maximum endurance of at least 4 hours. */
    }

    requirement <'3'> Req3_Velocity {
        doc /* The aircraft shall have a minimum velocity of at least 80 km/h. */
    }

    requirement <'4'> Req4_Thrust_Required {
        doc /* The aircraft propeller shall produce enough thrust to exceed drag. */
    }

    requirement <'5'> Req5_Controls {
        doc /* The aircraft shall determine accurate position and velocity data. */

        requirement <'5.1'> Req5_1_Sensors {
            doc /* The aircraft shall accurately determine altitude and speed. */
        }
        requirement <'5.2'> Req5_2_Navigation {
            doc /* The aircraft shall display accurate position and heading. */
        }
    }

    requirement <'6'> Req6_Stall_Velocity {
        doc /* The aircraft shall have a stall velocity of at least 80 km/h with flaps. */
    }

    <error>
    requirement <'7'> Req7_Maximum_Velocity_due_to_Thrust {
        doc /* The aircraft shall have sufficient thrust available to travel at a greater velocity than the stall velocity. */
    }
    </error>
    requirement <'8'> Req8_Pilot_Control {
        doc /* The pilot shall be able to control the heading of the aircraft based on displayed data. */
    }

    requirement <'9'> Req9_Stability {
        doc /* The aircraft shall have any features necessary for stable flight. */
    }			
}


//Parts Package
package Aircraft_Parts {
	part Aircraft {
        part avionicsSubsystem {
            part navigation;
            part sensor;
        }
        part controlsSubsystem {
            part aileron;
            part flap;
            part rudder;
        }
        <error>
        part propulsionSubsystem {
            part engine;
            part fuel;
            part propeller;
        }
        </error>
        part structuralSubsystem {
            part fuselage;
            part tail;
            part wing;
        }
    }
}

//Requirements Satisfaction Package
	package satisfactionContext {
		private import Aircraft_Requirements::*;
		private import Aircraft_Parts::*;
			
		satisfy Req5_Controls.Req5_2_Navigation by Aircraft.avionicsSubsystem.navigation;
		satisfy Req5_Controls.Req5_1_Sensors by Aircraft.avionicsSubsystem.sensor;
		satisfy Req8_Pilot_Control by Aircraft.controlsSubsystem.aileron;
		satisfy Req8_Pilot_Control by Aircraft.controlsSubsystem.flap;
		satisfy Req8_Pilot_Control by Aircraft.controlsSubsystem.rudder;
		satisfy Req9_Stability by Aircraft.structuralSubsystem.tail;
	}
}

