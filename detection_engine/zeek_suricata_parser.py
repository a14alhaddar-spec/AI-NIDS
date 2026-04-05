import json

def parse_suricata_alert(line):
    data = json.loads(line)
    features = [
        data["src_port"], data["dest_port"], data["proto"],
        data["flow"]["pkts_toserver"], data["flow"]["pkts_toclient"]
    ]
    return features
