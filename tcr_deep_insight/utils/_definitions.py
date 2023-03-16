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

study_name_palette = {'Abbas_2021': '#ffff00',
 'Azizi_2018': '#1ce6ff',
 'Bacher_2020': '#ff34ff',
 'Boland_2020': '#ff4a46',
 'Borcherding_2021': '#008941',
 'Cheon_2021': '#006fa6',
 'Corridoni_2020': '#a30059',
 'Gao_2020': '#ffdbe5',
 'Gate_2020': '#7a4900',
 'He_2020': '#0000a6',
 'Kim_2022': '#63ffac',
 'Krishna_2021': '#b79762',
 'Liao_2020': '#004d43',
 'Liu_2021': '#8fb0ff',
 'Lu_2019': '#997d87',
 'Luoma_2020': '#5a0007',
 'Mahuron_2020': '#809693',
 'Neal_2018': '#6a3a4c',
 'Notarbartolo_2021': '#1b4400',
 'Penkava_2020': '#4fc601',
 'Ramaswamy_2021': '#3b5dff',
 'Simone_2021': '#4a3b53',
 'Suo_2022': '#ff2f80',
 'Wang_2021': '#61615a',
 'Wang_2022': '#ba0900',
 'Wen_2020': '#6b7900',
 'Yost_2019': '#00c2a0',
 'Zheng_2020': '#ffaa92'}

reannotated_prediction_palette = {'CD4': '#1c83c5ff',
 'CD8': '#ffca39ff',
 'CD40LG': '#5bc8d9ff',
 'Cycling': '#a7a7a7ff',
 'MAIT': '#2a9d8fff',
 'Naive CD4': '#3c3354ff',
 'Naive CD8': '#a9d55dff',
 'Treg': '#6a4d93ff',
 'Undefined': '#f7f7f7ff',
 'Ambiguous': '#f7f7f7ff',
  'Unknown': '#f7f7f7ff'}

reannotation_palette = {'CREM+ Tm': '#1f77b4',
 'CXCR6+ Tex': '#ff7f0e',
 'Cycling T': '#279e68',
 'Early Tcm/Tem': '#d62728',
 'GZMK+ Tem': '#aa40fc',
 'GZMK+ Tex': '#8c564b',
 'IFITM3+KLRG1+ Temra': '#e377c2',
 'ILTCK': '#b5bd61',
 'ITGAE+ Trm': '#17becf',
 'ITGB2+ Trm': '#aec7e8',
 'KLRG1+ Temra': '#ffbb78',
 'KLRG1- Temra': '#98df8a',
 'MAIT': '#ff9896',
 'SELL+ progenitor Tex': '#c49c94',
 'Tcm': '#f7b6d2',
 'Tn': '#dbdb8d',
    'None': '#F7F7F7'
}
disease_type_palette = {'AML': '#E64B35',
 'COVID-19': '#4DBBD5',
 'Healthy': '#029F87',
 'Inflammation': '#3C5488',
 'Inflammation-irAE': '#F39B7F',
 'Solid tumor': '#8491B4',
 'T-LGLL': '#91D1C2'}
