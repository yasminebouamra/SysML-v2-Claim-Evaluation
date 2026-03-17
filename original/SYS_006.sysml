package Flashlight_StarterModel{

    private import ScalarValues::*; 
    private import SI::*;
    private import USCustomaryUnits::*;
    package FlashlightSpecificationAndDesign {
        package Requirements{
            requirement flashlightSpecification{
                requirement userInterface;
                requirement illumination{
                    requirement fieldOfView{
                        doc /* The flashlight field of view shall be 0 - 20 degrees.*/
                    }
                    requirement lightPower{
                        doc /* The light power shall be a minumum of TBD lumens.*/            
                    }
                }
                requirement physical{
                    requirement portability;
                    requirement size{
                        doc /* The flashlight shall be less than 6 inches in length.*/    
                    }
                    <error>
                    requirement weight{
                        doc /* The weight shall be less than 0.25 lbs.*/  
                        attribute actualWeight :> ISQ::mass;
                        require constraint {actualWeight <= 0.25 [lb]}
                    }
                    </error>
                }
                requirement reliability;
                requirement cost;
            }
        }

        package ActionTree{
            action produceDirectedLight{
                in item onOffCmd;
                out item lightOut;
                action provideDCPwr{
                    out item dcPwr;
                }
                action connectDCPwr {
                    in item onOffCmd;
                    in item dcPwrIn;
                    out item dcPwrOut;
                }
                action generateLight{
                    in item dcPwrIn;
                    out item light;
                }
                action directLight{
                    in item lightIn;
                    out item lightOut;
                }
                fork fork1;
                    then provideDCPwr;
                    then connectDCPwr;
                first start then fork1;
                bind onOffCmd = connectDCPwr.onOffCmd;
                bind lightOut=directLight.lightOut;
                flow from provideDCPwr.dcPwr to connectDCPwr.dcPwrIn;
                flow from connectDCPwr.dcPwrOut to generateLight.dcPwrIn;
                flow from generateLight.light to directLight.lightIn;
            }
        }


        package PartsTree{ 
            item def OnOff;
            item def DCPwr;
            item def Light;
            part flashlight {
                // flashlight attributes
                attribute mass :> ISQ::mass;
                attribute fov:Real;
                attribute illuminationLevel:Real;

                // flashlight ports
                <error>
                port cmdPort;
                port lightOutPort;
                port handPort;
                </error>
                perform ActionTree::produceDirectedLight;      

                exhibit state flashlightStates{
                    // declare the states
                    state initial;
                    state off;                                 
                    state on{
                        do produceDirectedLight;
                    }

                    transition initial then off;
                    transition off_To_on
                        first off
                        accept onOff:OnOff via cmdPort
                        then on;
                    transition on_To_off
                        first on
                        accept onOff:OnOff via cmdPort
                        then off;
                }

                ref part battery [2]{
                    attribute power:> ISQ::power;
                    port dcPwrOutPort{
                        out item dcPwr;
                    }
                    perform produceDirectedLight.provideDCPwr;
                }
                part switch{
                    port cmdPort;
                    port inPort{
                        in item dcPwrIn:DCPwr;
                    }
                    port outPort{
                        out item dcPwrOut:DCPwr;
                    }
                    perform produceDirectedLight.connectDCPwr;
                }
                <error>
                part lamp{
                    attribute efficiency:Real;
                    port dcPwrInPort{
                        in item dcPwrIn:DCPwr;
                    }
                    port lightOutPort{
                        out item light:Light;
                    }
                    perform produceDirectedLight.generateLight;
                }
                </error>
                part optics{
                    port lightInPort{
                        in item light:Light;
                    }
                    port lightOutPort{
                        out item directedLight:Light;
                    }
                    perform produceDirectedLight.directLight;
                    part reflector{
                        attribute radius:> ISQ::length;
                    }
                    part lens;
                }
                part structure{
                    port handPort;
                    part frontHousing;
                    part middleHousing;
                    part backHousing;
                }

                // the following establish connections
                // the binding connections assert that the cmdPort on the flashlight is the same as the cmdPort on the switch
                // and similarly for the lightOutPort
                bind cmdPort = switch.cmdPort;
                bind lightOutPort = optics.lightOutPort;
                bind handPort = structure.handPort;

                // the connections connect a port on one part to a port to on another part
                connection c1 connect battery.dcPwrOutPort to switch.inPort{
                    flow of DCPwr from source.dcPwr to target.dcPwrIn;
                }
                connect switch.outPort to lamp.dcPwrInPort{
                    flow of DCPwr from source.dcPwrOut to target.dcPwrIn;
                }
                connect lamp.lightOutPort to optics.lightInPort{
                    flow of Light from source.light to target.light;
                }
            }
        }
        // create an RequirementsAllocation package that contains the allocation of requirements to the flashlight design features
        package RequirementsAllocation{
            // this package imports the Requirements package and the PartsTree packge to access their elements
            // without having to use their fully qualified name such as Requirements::flashlightSpecification
            private import Requirements::*;
            private import PartsTree::*;

            // the following allocates the flashlightSpecification to the flashlight
            
            allocate flashlightSpecification to flashlight{
                allocate flashlightSpecification.physical.weight to flashlight.mass; // nested allocation
                allocate flashlightSpecification.illumination to flashlight.produceDirectedLight;
                allocate flashlightSpecification.illumination.fieldOfView to flashlight.fov;
                allocate flashlightSpecification.illumination.lightPower to flashlight.illuminationLevel;
            }
        }
    }