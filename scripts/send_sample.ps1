$body = @{
  src_ip = "10.0.0.10"
  dst_ip = "10.0.0.20"
  bytes = 9000000
  packets = 4000
  duration = 1.2
  src_bytes = 8500000
  dst_bytes = 500000
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:5001/ingest" -ContentType "application/json" -Body $body
