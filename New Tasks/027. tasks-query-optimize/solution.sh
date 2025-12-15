#!/bin/bash

# 1. Create Indices
sqlite3 /app/db/analytics.db "CREATE INDEX idx_logs_userid ON logs(user_id);"
sqlite3 /app/db/analytics.db "CREATE INDEX idx_logs_timestamp ON logs(timestamp);"
sqlite3 /app/db/analytics.db "CREATE INDEX idx_users_country ON users(country);"

# 2. Write Optimized Query
echo "SELECT count(*) 
FROM logs l
JOIN users u ON l.user_id = u.id
WHERE u.country = 'US'
  AND l.event_type = 'LOGIN'
  AND l.timestamp >= '2024-01-01' AND l.timestamp < '2024-02-01';" > /app/solution.sql