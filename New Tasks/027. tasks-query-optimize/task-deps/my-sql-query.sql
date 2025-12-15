-- Goal: Count 'LOGIN' events for 'US' users in 'January 2024'

SELECT count(*) 
FROM logs l
JOIN users u ON l.user_id = u.id
WHERE u.country = 'US'
  AND l.event_type = 'LOGIN'
  -- BAD PRACTICE: Using string manipulation on a datetime column prevents index usage
  AND l.timestamp LIKE '2024-01%';