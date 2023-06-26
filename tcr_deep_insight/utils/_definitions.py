from enum import Enum

class FCDEF(Enum):
    ENCODER = 0x0
    DECODER = 0x1

TRA_DEFINITION = ["CDR3a","TRAV","TRAJ"]
TRB_DEFINITION = ["CDR3b","TRBV","TRBJ"]
TRAB_DEFINITION = ["CDR3a","CDR3b","TRAV","TRAJ","TRBV","TRBJ"]


TRA_DEFINITION_ORIG = ["IR_VJ_1_junction_aa",
    "IR_VJ_1_v_call",
    "IR_VJ_1_j_call"
]
TRB_DEFINITION_ORIG = [
    "IR_VDJ_1_junction_aa",
    "IR_VDJ_1_v_call",
    "IR_VDJ_1_j_call"
]
TRAB_DEFINITION_ORIG = [
    "IR_VJ_1_junction_aa",            
    "IR_VDJ_1_junction_aa",
    "IR_VJ_1_v_call",
    "IR_VJ_1_j_call",
    "IR_VDJ_1_v_call",
    "IR_VDJ_1_j_call"
]