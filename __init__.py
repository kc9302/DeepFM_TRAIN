import torch
import logging
import configparser

# read config.ini
config = configparser.ConfigParser()
config.read("./config.ini")

field_dict = {
    0: ["mode_id"],
    1: ["grade_cd"],
    2: ["semstr_cd"],
    3: ["conts_dtl_qitem_type_se_cd"],
    4: ["conts_dtl_recmnd_hr_vl"]
}

field_index = [0, 1, 2, 3, 4]


# Operation flow sequence 1.
try:
    # setting device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"   
except Exception as err:
    logging.error(err)

