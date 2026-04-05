"""
Live Network Traffic Monitor
Captures packets from network interface and sends to inference services
Displays alerts on all dashboards in real-time
"""
import os
import time
import requests
from collections import defaultdict
from datetime import datetime
try:
    from scapy.all import sniff, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[WARN] Scapy not installed. Install with: pip install scapy")

# Configuration
DEFAULT_INFERENCE_HOST = os.getenv("INFERENCE_HOST", "localhost")
INFERENCE_SERVICES = {
    'Random Forest': os.getenv('INFERENCE_RF_URL', f'http://{DEFAULT_INFERENCE_HOST}:5002/infer'),
    'CNN-LSTM': os.getenv('INFERENCE_CNN_LSTM_URL', f'http://{DEFAULT_INFERENCE_HOST}:5003/infer'),
    'CNN': os.getenv('INFERENCE_CNN_URL', f'http://{DEFAULT_INFERENCE_HOST}:5004/infer'),
    'LSTM': os.getenv('INFERENCE_LSTM_URL', f'http://{DEFAULT_INFERENCE_HOST}:5005/infer')
}

# Flow tracking
flows = defaultdict(lambda: {
    'start_time': None,
    'packets_fwd': 0,
    'packets_bwd': 0,
    'bytes_fwd': 0,
    'bytes_bwd': 0,
    'src_ip': None,
    'dst_ip': None,
    'dst_port': None
})

# Statistics
alert_count = 0
packet_count = 0
flow_count = 0

def get_flow_key(pkt, is_fwd=True):
    """Generate flow key from packet"""
    if IP in pkt:
        if is_fwd:
            return f"{pkt[IP].src}-{pkt[IP].dst}-{pkt[TCP].sport if TCP in pkt else pkt[UDP].sport if UDP in pkt else 0}-{pkt[TCP].dport if TCP in pkt else pkt[UDP].dport if UDP in pkt else 0}"
        else:
            return f"{pkt[IP].dst}-{pkt[IP].src}-{pkt[TCP].dport if TCP in pkt else pkt[UDP].dport if UDP in pkt else 0}-{pkt[TCP].sport if TCP in pkt else pkt[UDP].sport if UDP in pkt else 0}"
    return None

def extract_features(flow_data):
    """Extract features from flow data for ML models"""
    duration = (time.time() - flow_data['start_time']) * 1000000 if flow_data['start_time'] else 1
    total_bytes = flow_data['bytes_fwd'] + flow_data['bytes_bwd']
    
    # Calculate flow bytes/sec (avoid division by zero)
    flow_bytes_sec = (total_bytes / (duration / 1000000)) if duration > 0 else 0
    
    features = {
        'Destination Port': flow_data['dst_port'] or 0,
        'Flow Duration': int(duration),
        'Total Fwd Packets': flow_data['packets_fwd'],
        'Total Backward Packets': flow_data['packets_bwd'],
        'Total Length of Fwd Packets': flow_data['bytes_fwd'],
        'Total Length of Bwd Packets': flow_data['bytes_bwd'],
        'Flow Bytes/s': flow_bytes_sec
    }
    
    return features

def send_to_inference(features, src_ip, dst_ip, dst_port):
    """Send features to all inference services and check for threats"""
    global alert_count
    
    threat_detected = False
    results = {}
    
    for model_name, url in INFERENCE_SERVICES.items():
        try:
            response = requests.post(url, json=features, timeout=2)
            if response.status_code == 200:
                result = response.json()
                results[model_name] = result
                
                if result.get('threat_detected', False):
                    threat_detected = True
        except:
            pass  # Service might not be running
    
    if threat_detected:
        alert_count += 1
        print(f"\n[ALERT] THREAT DETECTED!")
        print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Flow: {src_ip}:{dst_port} -> {dst_ip}")
        
        for model_name, result in results.items():
            if result.get('threat_detected'):
                threat_type = result.get('threat_type', 'Unknown')
                confidence = result.get('confidence', 0) * 100
                print(f"   {model_name}: {threat_type} ({confidence:.1f}% confidence)")

def process_packet(pkt):
    """Process captured packet"""
    global packet_count, flow_count
    
    if IP not in pkt:
        return
    
    packet_count += 1
    
    # Get flow key
    flow_key_fwd = get_flow_key(pkt, is_fwd=True)
    flow_key_bwd = get_flow_key(pkt, is_fwd=False)
    
    if not flow_key_fwd:
        return
    
    # Check if this is a new flow or reverse direction
    if flow_key_fwd in flows:
        flow = flows[flow_key_fwd]
        flow['packets_fwd'] += 1
        flow['bytes_fwd'] += len(pkt)
    elif flow_key_bwd in flows:
        flow = flows[flow_key_bwd]
        flow['packets_bwd'] += 1
        flow['bytes_bwd'] += len(pkt)
    else:
        # New flow
        flow_count += 1
        flow = flows[flow_key_fwd]
        flow['start_time'] = time.time()
        flow['src_ip'] = pkt[IP].src
        flow['dst_ip'] = pkt[IP].dst
        flow['dst_port'] = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport if UDP in pkt else 0
        flow['packets_fwd'] = 1
        flow['bytes_fwd'] = len(pkt)
    
    # Analyze flow every 10 packets or if it's suspicious
    total_packets = flow['packets_fwd'] + flow['packets_bwd']
    if total_packets >= 10 or flow['bytes_fwd'] > 10000:
        features = extract_features(flow)
        send_to_inference(features, flow['src_ip'], flow['dst_ip'], flow['dst_port'])

def get_network_interface():
    """Get available network interfaces"""
    try:
        from scapy.all import get_if_list
        interfaces = get_if_list()
        print("\n[INFO] Available Network Interfaces:")
        for i, iface in enumerate(interfaces, 1):
            print(f"   {i}. {iface}")
        return interfaces
    except:
        return []

def check_inference_services(verbose=True):
    """Return count of reachable inference services."""
    if verbose:
        print("\n[INFO] Checking inference services...")

    services_running = 0
    for model_name, url in INFERENCE_SERVICES.items():
        try:
            response = requests.get(url.replace('/infer', '/health'), timeout=2)
            if response.status_code == 200:
                if verbose:
                    print(f"   [OK] {model_name} service running")
                services_running += 1
            elif verbose:
                print(f"   [WARN] {model_name} service not responding")
        except:
            if verbose:
                print(f"   [ERROR] {model_name} service not accessible")

    return services_running

def main():
    print("\n" + "="*70)
    print("LIVE NETWORK TRAFFIC MONITOR")
    print("="*70)
    
    if not SCAPY_AVAILABLE:
        print("\n[ERROR] Scapy is required for packet capture")
        print("   Install: pip install scapy")
        print("\n   On Windows, you may also need:")
        print("   - Npcap: https://npcap.com/#download")
        return
    
    # Wait for at least one inference service so online mode can stay active.
    services_running = check_inference_services(verbose=True)
    while services_running == 0:
        print("\n[WARN] No inference services running yet.")
        print("   Start services with: .venv\\Scripts\\python.exe scripts\\launch_dashboards.py")
        print("   Retrying in 5 seconds...")
        time.sleep(5)
        services_running = check_inference_services(verbose=False)
        if services_running > 0:
            print("\n[INFO] Inference service detected. Continuing startup...")
    
    print(f"\n[OK] {services_running}/{len(INFERENCE_SERVICES)} services active")
    
    # Get network interface
    interfaces = get_network_interface()
    if not interfaces:
        print("\n[ERROR] No network interfaces found")
        return
    
    print("\n" + "="*70)
    print("CAPTURE CONFIGURATION")
    print("="*70)
    
    # Auto-select first non-loopback interface or let user choose
    default_iface = None
    for iface in interfaces:
        if 'loopback' not in iface.lower() and '127.0.0.1' not in iface:
            default_iface = iface
            break
    
    if not default_iface:
        default_iface = interfaces[0]
    
    print(f"\nUsing interface: {default_iface}")
    print(f"Monitoring traffic from Kali VM...")
    print(f"\nPress Ctrl+C to stop\n")
    
    print("="*70)
    print("LIVE MONITORING STARTED")
    print("="*70)
    
    try:
        # Start packet capture
        sniff(
            iface=default_iface,
            prn=process_packet,
            filter="tcp or udp",  # Capture TCP and UDP packets
            store=0  # Don't store packets in memory
        )
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("MONITORING STOPPED")
        print("="*70)
        print(f"\n[INFO] Statistics:")
        print(f"   Packets captured: {packet_count:,}")
        print(f"   Flows analyzed: {flow_count:,}")
        print(f"   Threats detected: {alert_count:,}")
        print(f"\n[OK] Session complete")
    except PermissionError:
        print("\n[ERROR] Permission denied!")
        print("   Run as Administrator to capture packets")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
