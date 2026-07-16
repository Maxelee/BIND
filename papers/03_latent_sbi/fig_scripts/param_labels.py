"""Curated short labels for the 30 SB35 astro parameter names.

Shared by fig01/fig06/fig08 (anywhere a full `param_names` string is too
long to fit as a tick label / legend entry). Replaces ad hoc fixed-width
`name[:N]` slicing, which cuts several long SB35 names mid-word (e.g.
'VariableWindVelFactor' -> 'VariableWindVelF'), leaving them unidentifiable
without cross-referencing an external table.

Each abbreviation below ends at a natural word/unit boundary (never
mid-word) and stays unambiguous against the other 29 names -- in
particular the three `WindEnergyReduction*` variants and the four
`BlackHole*Factor/Efficiency` variants remain distinguishable from each
other after abbreviation.

Usage:
    from param_labels import short_label
    ax.set_yticklabels([short_label(pn[i]) for i in top])
"""

# full SB35 name -> short, word-boundary-safe label
PARAM_ABBREV = {
    "WindEnergyIn1e51erg": "WindEnergy1e51erg",
    "RadioFeedbackFactor": "RadioFdbkFactor",
    "VariableWindVelFactor": "VarWindVelFactor",
    "RadioFeedbackReiorientationFactor": "RadioFdbkReorient",
    "MaxSfrTimescale": "MaxSfrTimescale",
    "FactorForSofterEQS": "FactorForSofterEQS",
    "IMFslope": "IMFslope",
    "SNII_MinMass_Msun": "SNII_MinMass_Msun",
    "ThermalWindFraction": "ThermalWindFrac",
    "VariableWindSpecMomentum": "VarWindSpecMom",
    "WindFreeTravelDensFac": "WindFreeTravelDens",
    "MinWindVel": "MinWindVel",
    "WindEnergyReductionFactor": "WindEReducFactor",
    "WindEnergyReductionMetallicity": "WindEReducMetal",
    "WindEnergyReductionExponent": "WindEReducExpon",
    "WindDumpFactor": "WindDumpFactor",
    "SeedBlackHoleMass": "SeedBlackHoleMass",
    "BlackHoleAccretionFactor": "BHAccretionFac",
    "BlackHoleEddingtonFactor": "BHEddingtonFac",
    "BlackHoleFeedbackFactor": "BHFeedbackFac",
    "BlackHoleRadiativeEfficiency": "BHRadiativeEff",
    "QuasarThreshold": "QuasarThreshold",
    "QuasarThresholdPower": "QuasarThreshPow",
    "UVBH0beta": "UVBH0beta",
    "UVBH0Deltaz": "UVBH0Deltaz",
    "UVBHepbeta": "UVBHepbeta",
    "UVBHepDeltaz": "UVBHepDeltaz",
    "SNIa_Rate_Norm": "SNIa_Rate_Norm",
    "SNIa_Rate_DTD_power": "SNIa_Rate_DTD_pow",
    "SofteningComovingType01": "SofteningComType01",
}


def short_label(name):
    """Curated abbreviation for `name`, or `name` unchanged if not long."""
    return PARAM_ABBREV.get(name, name)
