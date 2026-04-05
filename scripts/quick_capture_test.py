"""Quick test to capture packets from Kali VM"""

from scapy.all import *
import time

print("\n" + "="*70)
print("QUICK PACKET CAPTURE TEST")
print("="*70)

print("\nTarget: Packets from/to 192.168.197.128 (Kali)")
print("\nSniffing on ALL interfaces...")
print("Run an nmap scan from Kali now!\n")
print("Press Ctrl+C to stop\n")

packet_count = 0

def packet_handler(pkt):
    global packet_count
    packet_count += 1
    
    if IP in pkt:
        src = pkt[IP].src
        dst = pkt[IP].dst
        
        # Check if packet involves Kali IP
        if src == "192.168.197.128" or dst == "192.168.197.128":
            protocol = "TCP" if TCP in pkt else "UDP" if UDP in pkt else "Other"
            
            if TCP in pkt:
                port = pkt[TCP].dport if dst == "192.168.197.128" else pkt[TCP].sport
                print(f"✅ [{packet_count}] {protocol}: {src} → {dst}:{port}")
            else:
                print(f"✅ [{packet_count}] {protocol}: {src} → {dst}")

try:
    # Sniff with filter for Kali IP
    sniff(filter="host 192.168.197.128", prn=packet_handler, store=0)
except KeyboardInterrupt:
    print(f"\n\n✋ Stopped. Captured {packet_count} packets from/to Kali")
except Exception as e:
    print(f"\n❌ Error: {e}")
