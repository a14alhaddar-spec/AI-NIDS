"""Simple packet capture that works on Windows"""

from scapy.all import *
import sys

print("\n" + "="*70)
print("SIMPLE PACKET CAPTURE TEST")
print("="*70)

# List all interfaces with IPs
print("\n📡 Available Interfaces:")
try:
    import psutil
    addrs = psutil.net_if_addrs()
    interfaces_list = []
    
    for name, addr_list in addrs.items():
        for addr in addr_list:
            if addr.family == 2:  # AF_INET (IPv4)
                ip = addr.address
                if ip != '127.0.0.1':  # Skip localhost
                    interfaces_list.append((name, ip))
                    print(f"   {len(interfaces_list)}. {name} - {ip}")
    
    if not interfaces_list:
        print("   No network interfaces found!")
        sys.exit(1)
    
    # Auto-select interface with 192.168.197.1
    selected = None
    for name, ip in interfaces_list:
        if ip == "192.168.197.1":
            selected = (name, ip)
            print(f"\n✅ Auto-selected: {name} ({ip})")
            break
    
    if not selected:
        # Use first available
        selected = interfaces_list[0]
        print(f"\n⚠️  Using first interface: {selected[0]} ({selected[1]})")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

print(f"\nListening for packets from Kali (192.168.197.128)...")
print("Run nmap scan now from Kali!")
print("\nPress Ctrl+C to stop\n")

packet_count = 0
kali_packets = 0

def process_packet(pkt):
    global packet_count, kali_packets
    packet_count += 1
    
    # Check if packet has IP layer
    if IP in pkt:
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        
        # Look for Kali traffic
        if "192.168.197.128" in [src_ip, dst_ip]:
            kali_packets += 1
            
            if TCP in pkt:
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
                flags = pkt[TCP].flags
                print(f"✅ [{kali_packets}] TCP: {src_ip}:{sport} → {dst_ip}:{dport} [{flags}]")
            elif UDP in pkt:
                sport = pkt[UDP].sport
                dport = pkt[UDP].dport
                print(f"✅ [{kali_packets}] UDP: {src_ip}:{sport} → {dst_ip}:{dport}")
            elif ICMP in pkt:
                print(f"✅ [{kali_packets}] ICMP: {src_ip} → {dst_ip}")
            else:
                print(f"✅ [{kali_packets}] IP: {src_ip} → {dst_ip}")
    
    # Show stats every 100 packets
    if packet_count % 100 == 0:
        print(f"   [Stats: {packet_count} total, {kali_packets} from/to Kali]")

try:
    # Sniff WITHOUT filter - process all packets
    # This is more reliable on Windows
    sniff(prn=process_packet, store=0)
    
except KeyboardInterrupt:
    print(f"\n\n✋ Stopped!")
    print(f"   Total packets: {packet_count}")
    print(f"   Kali packets: {kali_packets}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTry running as Administrator if capture fails")
