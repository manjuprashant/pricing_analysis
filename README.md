# Hotel Booking Pricing Analysis

## Project Overview

This project analyzes hotel booking data to identify pricing drivers,
customer behavior patterns, and cancellation trends. The goal is to
support Revenue Management teams with data-driven recommendations for
dynamic pricing, customer retention, and revenue optimization.

## Business Problem

Hotels must balance occupancy, pricing, and customer retention while
minimizing revenue loss from cancellations. This project investigates
factors influencing Average Daily Rate (ADR), booking cancellations, and
customer segments to support strategic decision-making.

## Dataset

-   Dataset: `hotel_bookings.csv`
-   Records: 119,390 hotel bookings
-   Domain: Travel, Tourism, and Hospitality

## Project Objectives

-   Clean and preprocess hotel booking data.
-   Perform Exploratory Data Analysis (EDA).
-   Identify key pricing drivers.
-   Analyze customer cancellation behavior.
-   Perform statistical testing.
-   Create customer segments.
-   Build an interactive Power BI dashboard.
-   Deliver business recommendations.

## Data Cleaning & Feature Engineering

-   Handled missing values in `agent`, `company`, `children`, and
    `country`.
-   Removed duplicates.
-   Treated ADR outliers using the IQR method.
-   Created derived features for segmentation and analysis.

## Exploratory Data Analysis

Key areas analyzed: - Booking trends - ADR distribution - Hotel type
performance - Customer demographics - Market segments - Distribution
channels - Seasonal demand patterns

## Statistical Analysis

### Chi-Square Tests

Used to determine relationships between: - Hotel type and
cancellations - Market segment and cancellations

### T-Test

Used to compare ADR between canceled and non-canceled bookings.

## Pricing Drivers

Major variables influencing ADR: - Lead Time - Hotel Type - Arrival
Month - Arrival Week Number - Weekend Nights - Week Nights - Adults -
Children - Meal Type - Market Segment - Distribution Channel - Room
Type - Customer Type - Booking Changes

## Customer Segmentation

Customers were segmented based on: - Booking lead time - Stay duration -
Booking behavior - Repeat guest status

## Power BI Dashboard

Dashboard pages include: 1. Executive Overview 2. Pricing Analysis 3.
Customer Segmentation 4. Cancellation Analysis 5. Revenue Management
Insights

### Filters & Slicers

-   Hotel Type
-   Arrival Month
-   Market Segment
-   Customer Type
-   Country
-   ADR Range
-   Lead Time Range
-   Repeated Guest

## Key Findings

-   Longer lead times are associated with higher cancellation rates.
-   ADR varies significantly across seasons and market segments.
-   Repeat guests tend to cancel less frequently.
-   Certain room types and customer segments generate higher revenue.

## Strategic Recommendations

1.  Implement dynamic pricing based on demand and seasonality.
2.  Target high-value customer segments with personalized offers.
3.  Reduce cancellations through optimized booking policies.
4.  Monitor revenue KPIs through interactive dashboards.
5.  Continuously evaluate pricing drivers and customer behavior.

## Repository Structure

``` text
pricing_analysis/
├── archive/
├── notebooks/
├── dashboard/
├── sql_queries.sql
├── requirements.txt
└── README.md
```

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   SciPy
-   Jupyter Notebook
-   Power BI
-   GitHub

## Author

Manjula Srinivasan Data Analytics Intern -- Infotact Solutions
