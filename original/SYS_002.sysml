package breck_reqs {
  private import Requirements::*;

  requirement <id1> yourResponsibilityCode : RequirementCheck{
    doc /* Riders shall follow responsible basic ski and snowboard conduct in order to promote skier safety for all patrons of Breckenridge */
    requirement <id1_1> riderAndObjectAvoidence : DesignConstraintCheck {
      doc /* Riders shall be able to avoid those with the right of way or other objects under any riding circumstances */
      requirement <id1_1_1> control : DesignConstraintCheck {
        doc /* Riders shall always stay in control, and be able to stop or avoid other people or objects. */
      }
      <error>
      requirement <id1_1_2> rightOfWay : DesignConstraintCheck {
        doc /* Those riding ahead have the right of way and must be avoided by those trailing */
      }
      </error>
      requirement <id1_1_3> merging : DesignConstraintCheck {
        doc /* Merging riders shall look uphill and yield to others as to avoid them */
      }
    }
    requirement <id1_2> lackOfObstruction : DesignConstraintCheck {
      doc /* Riders shall not stop where they obstruct a trail or are not visible from above */
    }
    requirement <id1_3> observingWarnings : DesignConstraintCheck {
      doc /* Riders shall observe all posted signs and warnings, including those instructing to keep off closed trails and out of closed areas */
    }
  }

  requirement <id2> coloradoSkiSafetyAct : RequirementCheck {
    doc /* The skier and the ski area shall follow the responsibilities established by the Colorado Ski Safety Act */
    requirement <id2_1> skiSafetyLaw : DesignConstraintCheck {
      doc /* Skiers shall assume the risk of any injury to person or property resulting from any of the inherent dangers and risks of skiing and may not recover from any ski area operator for any injury resulting from any of the inherent dangers and risks of skiing */
      <error>requirement <id2_1_1> skierDefinition : DesignConstraintCheck {
        doc /* All patrons using the Breckenridge facility shall be acknowledged under law as a skier */
      }
      </error>
      requirement <id2_1_2> inherentRisks : DesignConstraintCheck {
        doc /* Skiers shall acknowledge that inherent risks include weather conditions, natural objects, man-made objects, and other skiers */
        requirement <id2_1_2_1> weatherRisks : DesignConstraintCheck {
          doc /* Inherent weather risks shall encompass changing temperature, changing snow conditions, and extraneous weather phenomena including avalanches */
        }
        requirement <id2_1_2_2> naturalObjectsRisks : DesignConstraintCheck {
          doc /* Natural object risks shall encompass bare spots, rocks, trees, and collisions with any such objects  */
        }
        requirement <id2_1_2_3> manMadeObjectsRisks : DesignConstraintCheck {
          doc /* Man-made object risks shall encompass any created jumps or terrains */
        }
        requirement <id2_1_2_4> otherSkiersRisks : DesignConstraintCheck {
          doc /* Other skiers risks shall encompass collisions with other skiers and injuries due to failure of skiing within ability levels */
        }
      }
    }
  }

  requirement <id3> smartStyleInFreestyleTerrain : RequirementCheck {
    doc /* Riders shall follow freestyle terrain ground rules to limit injuries caused to patrons in the freestyle terrain area */
    requirement <id3_1> featureUsage : DesignConstraintCheck {
      doc /* Only one rider shall use an open freestyle feature at any given time */
    }
    requirement <id3_2> featureInspection : DesignConstraintCheck {
      doc /* Riders shall be responsible for examining the freestyle feature before jumping */
    }
    requirement <id3_3> abilityLevel : DesignConstraintCheck {
      doc /* Riders shall ride within their ability level as this controls the difficulty of any freestyle feature */
    }
  }

  requirement <id4> liftSaftey : RequirementCheck {
    doc /* Riders boarding any lifts shall be physically capable, informed, and sober */
    <error>requirement <id4_1> riderBAC : PerformanceRequirementCheck {
      doc /* Riders shall have a blood alcohol content of 0.0 in order to use a lift */
    }
    </error>
    requirement <id4_2> physicalCapability : DesignConstraintCheck {
      doc /* Riders shall be able bodied and have physical dexterity  */
    }
    requirement <id4_3> riderKnowledge : DesignConstraintCheck {
      doc /* Riders shall be knowledgeable on safe operation of a lift or receive training on the matter before use */
    }
  }

  requirement def BACConstraintCheck :> RequirementCheck {
    subject: Part;
  }
  requirement <id5> BACdetection : BACConstraintCheck {
    doc /* The lift operator shall use a Breathalyzer on a rider with questionable sobriety to determine if they are sober */
  }

  part def Brethalizer;
  part <p1> brethalizer : Brethalizer {
    satisfy BACdetection;
  }
}
