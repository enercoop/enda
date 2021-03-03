# Electricity Load Forecast

This is a guide to get started with electricity load forecast using enda.

Some code is worth more than a thousand words. 

contracts_a : 
    * a list of contracts known on 2020-09-20 
    * each contract divided into sub-contracts : a sub-contract has fixed-values between two dates
    * a null end_date means the sub-contract is still active, and has no pre-determined end date
    * some contracts have a pre-determined end date in the future
    * some contracts are already signed but start in the future (see contract 6)





-> faire le squelette pour les utiliser

-> importer les fonctions pour le faire

-> ajouter les parties avancées comme la résilience à des données manquantes au moment du predict, à l'aide d'algos plus complexes

-> 



## Overview :

- desired output : 
  - the total load of a set of consumers 
  - every `timestep` : typically 15min, 30min or 1h.  
  - for the next few days, at least 1-2 days but typically 7 or 10 days.
- method : 
  1. divide the customers into meaningful groups (this could be just 1 big group)
  2. for each group:
    a. predict the load per customer (or per any summable quantity like subscribed power or annual expected consumption), per time-step in the next days. This step uses machine learning.
    b. predict the number of customers in this group, per time-step, in the next few days
    c. multiply : load per customer * number of customers at each time step
  3. sum the consumption of the different groups  

## Required data

For each group, you need the historical consumption with `timestep` granularity. 

You also need the corresponding `portfolio` over the past. This is a set of features describing how the "size" of the group evolves over time. 
For instance:
- the number of customers (more precisely : the number of active contracts, because some customer may have several contracts) 
- the sum of the subscribed power of these contracts over time (each contract's subscribed power may vary over time, but you may be able to ). 









