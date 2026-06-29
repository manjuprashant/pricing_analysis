 # TOTAL REVENUE

SELECT SUM(adr * 
(stays_in_weekend_nights +
stays_in_week_nights))
AS total_revenue
FROM hotel_bookings;


# Cancellation Rate

SELECT
ROUND(
100.0 * SUM(is_canceled)
/ COUNT(*),
2
) AS cancellation_rate
FROM hotel_bookings;



# 

SELECT AVG(adr)
AS average_adr
FROM hotel_bookings;



# Monthly Demand

SELECT
arrival_date_month,
COUNT(*) bookings
FROM hotel_bookings
GROUP BY arrival_date_month
ORDER BY bookings DESC;
