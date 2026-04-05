"""Check actual column names in CICIDS CSV"""

import pandas as pd

file_path = r"datasets\CICIDS2017\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
df = pd.read_csv(file_path, nrows=1)

print("\nAll column names:")
print("="*70)
for i, col in enumerate(df.columns, 1):
    print(f"{i:3d}. '{col}'")

print("\n\nLooking for our 7 features:")
print("="*70)
needed = ['Destination Port', 'Flow Duration', 'Total Fwd Packets',
          'Total Backward Packets', 'Fwd Packet', 'Bwd Packet', 'Flow Bytes']

for need in needed:
    matches = [col for col in df.columns if need.lower() in col.lower()]
    print(f"\nSearching for '{need}':")
    for match in matches:
        print(f"  → '{match}'")
