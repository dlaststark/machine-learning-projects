select 
	id, price, levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags, count(1)
FROM machinehack.train_data
group by 
	id, price, levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
having count(1) > 1
order by id; --220

select 
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags, count(1)
FROM machinehack.train_data
group by 
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
having count(1) > 1
order by
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags;

select distinct 
	price, levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
from machinehack.train_data
where levy = '640' and manufacturer = 'FORD' and model = 'Fusion'
and "Prod. year" = 2013 and category = 'Sedan' and mileage = '174619 km'
and color = 'Grey' order by price;

select count(1) from machinehack.train_data; --19237

select count(1) from (
select distinct 
	id, price, levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
from machinehack.train_data) base; -- 15725


select 
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags, count(1)
FROM (
select distinct 
	id, price, levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
from machinehack.train_data) base
group by 
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
having count(1) > 1
order by
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags;


select distinct 
	id, price, levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
from machinehack.train_data
where levy = '1017' and manufacturer = 'MERCEDES-BENZ' and model = 'E 300'
and "Prod. year" = 2017 and category = 'Sedan' and mileage = '26802 km'
and color = 'Black' order by price;


select 
	avg(price) as mean_price, sum(price) as sum_price,
	--PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY price) as median_price,
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
from (
select distinct 
	price, levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
from machinehack.train_data
where levy = '640' and manufacturer = 'AUDI' and model = 'A4'
and "Prod. year" = 2013 and category = 'Sedan' and mileage = '157584 km'
and color = 'Black'
) base
group by
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
order by 1;



with train as 
(	
	select 
		levy, manufacturer, model, "Prod. year", category, 
		"Leather interior", "Fuel type", "Engine volume", mileage, 
		cylinders, "Gear box type", "Drive wheels", doors, wheel, 
		color, airbags, count(1)
	FROM (
	select distinct 
		price, levy, manufacturer, model, "Prod. year", category, 
		"Leather interior", "Fuel type", "Engine volume", mileage, 
		cylinders, "Gear box type", "Drive wheels", doors, wheel, 
		color, airbags
	from machinehack.train_data) base
	group by 
		levy, manufacturer, model, "Prod. year", category, 
		"Leather interior", "Fuel type", "Engine volume", mileage, 
		cylinders, "Gear box type", "Drive wheels", doors, wheel, 
		color, airbags
	having count(1) < 2
)
select count(1)
from train join machinehack.test_data test
on  train.levy = test.levy
and train.manufacturer = test.manufacturer
and train.model = test.model
and train."Prod. year" = test."Prod. year"
and train.category = test.category
and train."Leather interior" = test."Leather interior"
and train."Fuel type" = test."Fuel type"
and train."Engine volume" = test."Engine volume"
and train.mileage = test.mileage
and train.cylinders = test.cylinders
and train."Gear box type" = test."Gear box type"
and train."Drive wheels" = test."Drive wheels"
and train.doors = test.doors
and train.wheel = test.wheel
and train.color = test.color
and train.airbags = test.airbags;  --1902


with train as 
(	
	select distinct 
		price, levy, manufacturer, model, "Prod. year", category, 
		"Leather interior", "Fuel type", "Engine volume", mileage, 
		cylinders, "Gear box type", "Drive wheels", doors, wheel, 
		color, airbags
	from machinehack.train_data
)

select airbags, count(1)
from train
where not(manufacturer in ('SATURN','HAVAL','GREATWALL','PONTIAC','LANCIA','LAMBORGHINI','ASTON MARTIN','TESLA')
or "Prod. year" in (1981,1943,1947,1982,1957,1968,1973,1974,1976)
or "Fuel type" = 'Hydrogen'
or "Engine volume" in ('20','6.7','0.4 Turbo','5.7 Turbo','5.8','0.2 Turbo','5.4 Turbo','7.3','0.8 Turbo','0.5','3.1','1.1 Turbo','6.8','5.2','0.3 Turbo')
or cylinders in (9, 14)
)
group by 1 order by 2;

select * from machinehack.train_data where manufacturer in ('SATURN','GREATWALL','LAMBORGHINI','ASTON MARTIN','TESLA');
select * from machinehack.test_data where manufacturer in ('SATURN','GREATWALL','LAMBORGHINI','ASTON MARTIN','TESLA');

select * from machinehack.train_data where "Prod. year" in (1981,1943,1968,1974) order by "Prod. year";
select * from machinehack.test_data where "Prod. year" in (1981,1943,1968,1974) order by "Prod. year";

select * from machinehack.train_data where "Fuel type" = 'Hydrogen';
select * from machinehack.test_data where "Fuel type" = 'Hydrogen';

select distinct * from machinehack.train_data where "Engine volume" in ('20','6.7','0.4 Turbo','5.7 Turbo','5.8','0.2 Turbo','5.4 Turbo','7.3','0.8 Turbo','0.5','3.1','1.1 Turbo','6.8','5.2','0.3 Turbo') order by "Engine volume";
select distinct * from machinehack.test_data where "Engine volume" in ('20','6.7','0.4 Turbo','5.7 Turbo','5.8','0.2 Turbo','5.4 Turbo','7.3','0.8 Turbo','0.5','3.1','1.1 Turbo','6.8','5.2','0.3 Turbo') order by "Engine volume";

select * from machinehack.train_data where cylinders in (9, 14);
select * from machinehack.test_data where cylinders in (9, 14);


select * from
(select cylinders, count(1) train_count from machinehack.train_data group by 1) train
full outer join (select cylinders, count(1) test_count from machinehack.test_data group by 1) test
on train.cylinders = test.cylinders order by 1,3;


with base as (
select distinct train.*
from (
	select distinct 
		price, levy, manufacturer, model, "Prod. year", category, 
		"Leather interior", "Fuel type", "Engine volume", mileage, 
		cylinders, "Gear box type", "Drive wheels", doors, wheel, 
		color, airbags--, id
	from machinehack.train_data) train
join (
	select distinct 
		price, levy, manufacturer, model, "Prod. year", category, 
		"Leather interior", "Fuel type", "Engine volume", mileage, 
		cylinders, "Gear box type", "Drive wheels", doors, wheel, 
		color, airbags--, id
	from machinehack.test_data) test
on  train.levy = test.levy
and train.manufacturer = test.manufacturer
and train.model = test.model
and train."Prod. year" = test."Prod. year"
and train.category = test.category
and train."Leather interior" = test."Leather interior"
and train."Fuel type" = test."Fuel type"
and train."Engine volume" = test."Engine volume"
and train.mileage = test.mileage
and train.cylinders = test.cylinders
and train."Gear box type" = test."Gear box type"
and train."Drive wheels" = test."Drive wheels"
and train.doors = test.doors
and train.wheel = test.wheel
and train.color = test.color
and train.airbags = test.airbags
--and train.id = test.id
)

select count(1) from (
select distinct 
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
from base) tmp;

select
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags, count(1)
from base 
group by
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
having count(1) > 1
order by
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags;


select distinct 'TRAIN' as source, 
	price, levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags, id
from machinehack.train_data 
where levy = '1017' and manufacturer = 'MERCEDES-BENZ' and model = 'GLA 250'
and "Prod. year" = 2017 and category = 'Jeep' and mileage = '44995 km'
and color = 'Black'
UNION
select distinct 'TEST' as source, 
	0 as price, levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags, id
from machinehack.test_data 
where levy = '1017' and manufacturer = 'MERCEDES-BENZ' and model = 'GLA 250'
and "Prod. year" = 2017 and category = 'Jeep' and mileage = '44995 km'
and color = 'Black';


select distinct price, levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
from machinehack.train_data;


select count(1) from (
select distinct *
from machinehack.test_data test left join machinehack.train_data train
on  train.levy = test.levy
and train.manufacturer = test.manufacturer
and train.model = test.model
and train."Prod. year" = test."Prod. year"
and train.category = test.category
and train."Leather interior" = test."Leather interior"
and train."Fuel type" = test."Fuel type"
and train."Engine volume" = test."Engine volume"
and train.mileage = test.mileage
and train.cylinders = test.cylinders
and train."Gear box type" = test."Gear box type"
and train."Drive wheels" = test."Drive wheels"
and train.doors = test.doors
and train.wheel = test.wheel
and train.color = test.color
and train.airbags = test.airbags
--and train.id = test.id
) base;


with base as (
select 
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags, count(1)
from (
select distinct 
	price, levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
from machinehack.train_data) base
group by
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
having count(1) > 1)

select 
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags, count(1)
from 
(select distinct
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
from machinehack.test_data) test
where (levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags) in
(select 
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags
from base)
group by
	levy, manufacturer, model, "Prod. year", category, 
	"Leather interior", "Fuel type", "Engine volume", mileage, 
	cylinders, "Gear box type", "Drive wheels", doors, wheel, 
	color, airbags;
--having count(1) > 1;

