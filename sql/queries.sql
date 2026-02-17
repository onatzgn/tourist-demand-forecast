-- 1) top landmarks by visits
SELECT city, landmark_id, COUNT(*) AS total_visits
FROM visits
GROUP BY city, landmark_id
ORDER BY total_visits DESC
LIMIT 20;

-- 2) seasonality by city (monthly demand)
SELECT city,
       CAST(strftime('%m', visit_timestamp) AS INTEGER) AS month,
       COUNT(*) AS monthly_visits
FROM visits
GROUP BY city, month
ORDER BY city, month;

-- 3) weekend vs weekday ratios
SELECT city,
       SUM(CASE WHEN strftime('%w', visit_timestamp) IN ('0','6') THEN 1 ELSE 0 END) AS weekend_visits,
       SUM(CASE WHEN strftime('%w', visit_timestamp) NOT IN ('0','6') THEN 1 ELSE 0 END) AS weekday_visits,
       ROUND(1.0 * SUM(CASE WHEN strftime('%w', visit_timestamp) IN ('0','6') THEN 1 ELSE 0 END) /
             NULLIF(SUM(CASE WHEN strftime('%w', visit_timestamp) NOT IN ('0','6') THEN 1 ELSE 0 END), 0), 4) AS weekend_weekday_ratio
FROM visits
GROUP BY city
ORDER BY weekend_weekday_ratio DESC;

-- 4) ticket price vs demand proxy
SELECT m.city,
       m.landmark_id,
       m.ticket_price,
       m.popularity_base_score,
       COUNT(v.visit_id) AS visits_count
FROM landmark_meta m
LEFT JOIN visits v ON v.landmark_id = m.landmark_id AND v.city = m.city
GROUP BY m.city, m.landmark_id, m.ticket_price, m.popularity_base_score
ORDER BY m.city, visits_count DESC;

-- 5) returning user impact
SELECT city,
       is_returning_user,
       COUNT(*) AS visits,
       ROUND(AVG(session_duration_sec), 2) AS avg_session_duration,
       ROUND(AVG(party_size), 2) AS avg_party_size
FROM visits
GROUP BY city, is_returning_user
ORDER BY city, is_returning_user DESC;

-- 6) referral mix over time
SELECT city,
       strftime('%Y-%m', visit_timestamp) AS ym,
       referral_source,
       COUNT(*) AS visits
FROM visits
GROUP BY city, ym, referral_source
ORDER BY city, ym, visits DESC;

-- 7) city-level demand trend (weekly)
SELECT city,
       date(visit_timestamp, '-' || ((strftime('%w', visit_timestamp) + 6) % 7) || ' days') AS week_start,
       COUNT(*) AS weekly_visits
FROM visits
GROUP BY city, week_start
ORDER BY city, week_start;

-- 8) event impact proxy (daily)
WITH daily_visits AS (
    SELECT city, date(visit_timestamp) AS date, COUNT(*) AS daily_visit_count
    FROM visits
    GROUP BY city, date(visit_timestamp)
),
daily_events AS (
    SELECT city, date, 1 AS has_event, MAX(event_intensity) AS max_event_intensity
    FROM events
    GROUP BY city, date
)
SELECT dv.city,
       COALESCE(de.has_event, 0) AS has_event,
       ROUND(AVG(dv.daily_visit_count), 2) AS avg_daily_visits,
       ROUND(AVG(COALESCE(de.max_event_intensity, 0)), 2) AS avg_event_intensity,
       COUNT(*) AS days_count
FROM daily_visits dv
LEFT JOIN daily_events de ON dv.city = de.city AND dv.date = de.date
GROUP BY dv.city, has_event
ORDER BY dv.city, has_event DESC;
