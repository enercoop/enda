# Electricity Load Forecast

This is a guide to get started with electricity load forecast using enda.

Briefly :
- desired output : 
  - the total load of a set of consumers 
  - every 1h, or another time-step <1h, like 15min or 30min
  - for the next few days, at least 1-2 days but typically 7 or 10 days.
- method : 
  - divide the customers into meaningful groups
    - this could be just 1 big group
  - for each group:
    - predict the load per customer
      (or per any summable quantity like subscribed power or annual expected consumption),
      per time-step in the next days
    - predict the number of customers in this group, per time-step, in the next few days
    - multiply : load per customer * number of customers at each time step
  - sum the consumption of the different groups  
- 


## Output : 

## Modelling customers : contracts data

## Output data: 






## Input and output data


