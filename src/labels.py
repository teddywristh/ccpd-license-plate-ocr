from pathlib import Path

PROVINCE_NAMES = [
    "Anhui", "Shanghai", "Tianjin", "Chongqing", "Hebei",
    "Shanxi", "Inner Mongolia", "Liaoning", "Jilin",
    "Heilongjiang", "Jiangsu", "Zhejiang", "Beijing",
    "Fujian", "Jiangxi", "Shandong", "Henan", "Hubei",
    "Hunan", "Guangdong", "Guangxi", "Hainan", "Sichuan",
    "Guizhou", "Yunnan", "Tibet", "Shaanxi", "Gansu",
    "Qinghai", "Ningxia", "Xinjiang", "Police", "Student",
    "No-character",
]
PROVINCE_TOKENS = (
    [f"P{i:02d}" for i in range(31)]
    + ["POLICE", "STUDENT", "NONE"]
)

LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
ALPHABET_TABLE = LETTERS + ["NONE"]
ADS_TABLE = LETTERS + list("0123456789") + ["NONE"]
CLASSES = PROVINCE_TOKENS + LETTERS + list("0123456789") + ["NONE"]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
TOKEN_TO_ID = {
    token: index + 3
    for index, token in enumerate(CLASSES)
}
ID_TO_TOKEN = {
    index: token
    for token, index in TOKEN_TO_ID.items()
}
def enc_label(path):
    fields = Path(path).stem.split("-")
    if len(fields) != 7:
        raise ValueError(f"Invalid CCPD filename: {path}")
    ids = [int(x) for x in fields[4].split("_")]
    if len(ids) != 8:
        raise ValueError(f"Green label needs 8 IDs, got {len(ids)}: {path}")
    tokens = [PROVINCE_TOKENS[ids[0]], ALPHABET_TABLE[ids[1]], *[ADS_TABLE[i] for i in ids[2:]]]
    tokens = [token for token in tokens if token != "NONE"]
    return [TOKEN_TO_ID[token] for token in tokens]

def show_pred(ids):
    tokens = [ID_TO_TOKEN[i] for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
    if tokens and tokens[0].startswith("P") and tokens[0][1:].isdigit():
        p_idx = int(tokens[0][1:])
        return PROVINCE_NAMES[p_idx] + " " + "".join(tokens[1:])
    return "".join(tokens)

def par_bbox(path):
    fields = Path(path).stem.split("-")
    if len(fields) != 7:
        raise ValueError(f"Invalid CCPD filename: {path}")
    top_left, bottom_right = fields[2].split("_")
    x1, y1 = map(int, top_left.split("&"))
    x2, y2 = map(int, bottom_right.split("&"))
    return x1, y1, x2, y2
