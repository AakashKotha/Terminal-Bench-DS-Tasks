SELECT count(*) 
FROM logs l
JOIN users u ON l.user_id = u.id
WHERE u.country = 'US'
  AND l.event_type = 'LOGIN'
  AND l.timestamp >= '2024-01-01' 
  AND l.timestamp < '2024-02-01';