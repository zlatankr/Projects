# -*- coding: utf-8 -*-
"""
Created on Sat Jul 01 17:37:49 2017

@author: User
"""
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://datascisba@c4sf-sba:9th&howard!!@c4sf-sba.postgres.database.azure.com:5432/postgres')

table_names = """
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'data_ingest'
  AND table_name   = 'irs_zip_data'
"""


sba_sql = """
drop table if exists stg_analytics.sba_sfdo;
create table stg_analytics.sba_sfdo

(
"Program"	varchar
,"BorrName"	text
,"BorrStreet"	text
,"BorrCity"	text
,"BorrState"	text
,"BorrZip"	bigint
,"GrossApproval"	bigint
,"ApprovalDate"	timestamp without time zone
,"ApprovalFiscalYear"	bigint
,"FirstDisbursementDate"	timestamp without time zone
,"DeliveryMethod"	text
,"subpgmdesc"	text
,"InitialInterestRate"	double precision
,"TermInMonths"	bigint
,"NaicsCode"	double precision
,"NaicsDescription"	text
,"FranchiseCode"	bigint
,"FranchiseName"	text
,"ProjectCounty"	text
,"ProjectState"	text
,"SBADistrictOffice"	text
,"CongressionalDistrict"	double precision
,"BusinessType"	text
,"LoanStatus"	text
,"ChargeOffDate"	timestamp without time zone
,"GrossChargeOffAmount"	bigint
,"JobsSupported"	bigint
,"CDC_Name"	text
,"CDC_Street"	text
,"CDC_City"	text
,"CDC_State"	text
,"CDC_Zip"	double precision
,"ThirdPartyLender_Name"	text
,"ThirdPartyLender_City"	text
,"ThirdPartyLender_State"	text
,"ThirdPartyDollars"	double precision
,"BankName"	text
,"BankStreet"	text
,"BankCity"	text
,"BankState"	text
,"BankZip"	text
,"SBAGuaranteedApproval"	bigint
,"RevolverStatus"	 bigint
);

insert into stg_analytics.sba_sfdo

select

CAST("Program" as varchar) "Program"
,"BorrName"
,"BorrStreet"
,"BorrCity"
,"BorrState"
,"BorrZip"
,"GrossApproval"
,"ApprovalDate"
,"ApprovalFiscalYear"
,"FirstDisbursementDate"
,"DeliveryMethod"
,"subpgmdesc"
,"InitialInterestRate"
,"TermInMonths"
,"NaicsCode"
,"NaicsDescription"
,"FranchiseCode"
,"FranchiseName"
,"ProjectCounty"
,"ProjectState"
,"SBADistrictOffice"
,"CongressionalDistrict"
,"BusinessType"
,"LoanStatus"
,"ChargeOffDate"
,"GrossChargeOffAmount"
,"JobsSupported"
,"CDC_Name"
,"CDC_Street"
,"CDC_City"
,"CDC_State"
,"CDC_Zip"
,"ThirdPartyLender_Name"
,"ThirdPartyLender_City"
,"ThirdPartyLender_State"
,"ThirdPartyDollars"
,null as "BankName"
,null as "BankStreet"
,null as "BankCity"
,null as "BankState"
,null as "BankZip"
,null as "SBAGuaranteedApproval"
,null as "RevolverStatus"

from data_ingest.sba__foia_504_1991_present

where 
"BorrState" = 'CA'
and
"ProjectCounty" in ('SANTA CRUZ', 'SANTA CLARA', 'SAN MATEO', 
                        'ALAMEDA', 'CONTRA COSTA', 'MARIN',
                        'SAN FRANCISCO', 'SOLANO', 'NAPA', 'SONOMA', 
                        'LAKE', 'MENDOCINO', 'HUMBOLDT', 'DEL NORTE')

union

select

CAST("Program" as varchar) "Program"
,"BorrName"
,"BorrStreet"
,"BorrCity"
,"BorrState"
,"BorrZip"
,"GrossApproval"
,"ApprovalDate"
,"ApprovalFiscalYear"
,"FirstDisbursementDate"
,"DeliveryMethod"
,"subpgmdesc"
,"InitialInterestRate"
,"TermInMonths"
,"NaicsCode"
,"NaicsDescription"
,"FranchiseCode"
,"FranchiseName"
,"ProjectCounty"
,"ProjectState"
,"SBADistrictOffice"
,"CongressionalDistrict"
,"BusinessType"
,"LoanStatus"
,"ChargeOffDate"
,"GrossChargeOffAmount"
,"JobsSupported"
,null as "CDC_Name"
,null as "CDC_Street"
,null as "CDC_City"
,null as "CDC_State"
,null as "CDC_Zip"
,null as "ThirdPartyLender_Name"
,null as "ThirdPartyLender_City"
,null as "ThirdPartyLender_State"
,null as "ThirdPartyDollars"
,"BankName"
,"BankStreet"
,"BankCity"
,"BankState"
,"BankZip"
,"SBAGuaranteedApproval"
,"RevolverStatus"

from data_ingest.sba__foia_7a_1991_1999

where 
"BorrState" = 'CA'
and
"ProjectCounty" in ('SANTA CRUZ', 'SANTA CLARA', 'SAN MATEO', 
                        'ALAMEDA', 'CONTRA COSTA', 'MARIN',
                        'SAN FRANCISCO', 'SOLANO', 'NAPA', 'SONOMA', 
                        'LAKE', 'MENDOCINO', 'HUMBOLDT', 'DEL NORTE')

union

select

CAST("Program" as varchar) "Program"
,"BorrName"
,"BorrStreet"
,"BorrCity"
,"BorrState"
,"BorrZip"
,"GrossApproval"
,"ApprovalDate"
,"ApprovalFiscalYear"
,"FirstDisbursementDate"
,"DeliveryMethod"
,"subpgmdesc"
,"InitialInterestRate"
,"TermInMonths"
,"NaicsCode"
,"NaicsDescription"
,"FranchiseCode"
,"FranchiseName"
,"ProjectCounty"
,"ProjectState"
,"SBADistrictOffice"
,"CongressionalDistrict"
,"BusinessType"
,"LoanStatus"
,"ChargeOffDate"
,"GrossChargeOffAmount"
,"JobsSupported"
,null as "CDC_Name"
,null as "CDC_Street"
,null as "CDC_City"
,null as "CDC_State"
,null as "CDC_Zip"
,null as "ThirdPartyLender_Name"
,null as "ThirdPartyLender_City"
,null as "ThirdPartyLender_State"
,null as "ThirdPartyDollars"
,"BankName"
,"BankStreet"
,"BankCity"
,"BankState"
,"BankZip"
,"SBAGuaranteedApproval"
,"RevolverStatus"

from data_ingest.sba__foia_7a_2000_2009

where 
"BorrState" = 'CA'
and
"ProjectCounty" in ('SANTA CRUZ', 'SANTA CLARA', 'SAN MATEO', 
                        'ALAMEDA', 'CONTRA COSTA', 'MARIN',
                        'SAN FRANCISCO', 'SOLANO', 'NAPA', 'SONOMA', 
                        'LAKE', 'MENDOCINO', 'HUMBOLDT', 'DEL NORTE')

union

select

CAST("Program" as varchar) "Program"
,"BorrName"
,"BorrStreet"
,"BorrCity"
,"BorrState"
,"BorrZip"
,"GrossApproval"
,"ApprovalDate"
,"ApprovalFiscalYear"
,"FirstDisbursementDate"
,"DeliveryMethod"
,"subpgmdesc"
,"InitialInterestRate"
,"TermInMonths"
,"NaicsCode"
,"NaicsDescription"
,"FranchiseCode"
,"FranchiseName"
,"ProjectCounty"
,"ProjectState"
,"SBADistrictOffice"
,"CongressionalDistrict"
,"BusinessType"
,"LoanStatus"
,"ChargeOffDate"
,"GrossChargeOffAmount"
,"JobsSupported"
,null as "CDC_Name"
,null as "CDC_Street"
,null as "CDC_City"
,null as "CDC_State"
,null as "CDC_Zip"
,null as "ThirdPartyLender_Name"
,null as "ThirdPartyLender_City"
,null as "ThirdPartyLender_State"
,null as "ThirdPartyDollars"
,"BankName"
,"BankStreet"
,"BankCity"
,"BankState"
,"BankZip"
,"SBAGuaranteedApproval"
,"RevolverStatus"

from data_ingest.sba__foia_7a_2010_present

where 
"BorrState" = 'CA'
and
"ProjectCounty" in ('SANTA CRUZ', 'SANTA CLARA', 'SAN MATEO', 
                        'ALAMEDA', 'CONTRA COSTA', 'MARIN',
                        'SAN FRANCISCO', 'SOLANO', 'NAPA', 'SONOMA', 
                        'LAKE', 'MENDOCINO', 'HUMBOLDT', 'DEL NORTE')
"""

naics_sql = """
drop table if exists stg_analytics.census_naics;
create table stg_analytics.census_naics

(
"ZIPCODE" varchar
,"GEO_ID" varchar
,"NAICS2012" varchar
,"NAICS2012_TTL" varchar
,"ESTAB" varchar);

insert into stg_analytics.census_naics

select

"ZIPCODE"
,"GEO_ID"
,"NAICS2012"
,"NAICS2012_TTL"
,sum(cast("ESTAB" as int))

from data_ingest.census__zip_business_patterns

where 

"EMPSZES_TTL" in ('Establishments with 1 to 4 employees',
             'Establishments with 5 to 9 employees',
             'Establishments with 10 to 19 employees',
             'Establishments with 20 to 49 employees',
             'Establishments with 50 to 99 employees',
             'Establishments with 100 to 249 employees',
             'Establishments with 250 to 499 employees')

and cast("ZIPCODE" as varchar) in (select distinct cast("BorrZip" as varchar) from stg_analytics.sba_sfdo)

group by 

"ZIPCODE"
,"GEO_ID"
,"NAICS2012"
,"NAICS2012_TTL"
"""

irs_sql = """

drop table if exists stg_analytics.irs_income;
create table stg_analytics.irs_income

(
"ZIPCODE" varchar
,"MEAN_AGI" varchar
);

insert into stg_analytics.irs_income

select 

"ZIPCODE"
,ceiling((sum("A00100")/sum("N1" + "MARS2"))*1000) as "MEAN_AGI"

from data_ingest.irs_zip_data

group by "ZIPCODE"

"""

combined = """
drop table if exists trg_analytics.sba_zip_level;
create table trg_analytics.sba_zip_level

(
"BorrZip" bigint
,"Total_SBA" int
,"Total_504" int
,"Total_7A" int
,"Total_Small_Bus" varchar
,"Mean_Agi" varchar
,"SBA_per_Small_Bus" float
,"504_per_Small_Bus" float
,"7A_per_Small_Bus" float);

insert into trg_analytics.sba_zip_level
select

A_combined."BorrZip"
,A_combined."Total_SBA"
,A_504."Total_504"
,A_7A."Total_7A"
,A_naics."ESTAB" as "Total_Small_Bus"
,A_irs."MEAN_AGI" as "Mean_Agi"
,cast(A_combined."Total_SBA" as float)/cast(A_naics."ESTAB" as float) as "SBA_per_Small_Bus"
,cast(A_504."Total_504" as float)/cast(A_naics."ESTAB" as float) as "504_per_Small_Bus"
,cast(A_7A."Total_7A" as float)/cast(A_naics."ESTAB" as float) as "7A_per_Small_Bus"

from 

(select

"BorrZip"
,count(*) as "Total_SBA"

from stg_analytics.sba_sfdo

group by "BorrZip") A_combined

left join

(select

"BorrZip"
,count(*) as "Total_504"

from stg_analytics.sba_sfdo

where "Program" = '504'

group by "BorrZip") A_504

on A_combined."BorrZip" = A_504."BorrZip"

left join

(select

"BorrZip"
,count(*) as "Total_7A"

from stg_analytics.sba_sfdo

where "Program" = '7A'

group by "BorrZip") A_7A

on A_combined."BorrZip" = A_7A."BorrZip"

left join

(select

"ESTAB"
,"GEO_ID"
,cast("ZIPCODE" as bigint)

from stg_analytics.census_naics

where "NAICS2012" = '00') A_naics

on A_combined."BorrZip" = A_naics."ZIPCODE"

left join

(select

cast("ZIPCODE" as bigint)
,"MEAN_AGI"

from stg_analytics.irs_income) A_irs

on A_combined."BorrZip" = A_irs."ZIPCODE"

"""