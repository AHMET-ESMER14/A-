Select * from Products

Select ProductName,ProductId,UnitPrice from Products

Select ProductName,ProductId,UnitPrice, * from Products

-- order by 
Select * from Products order by ProductName

Select * from Products order by ProductName desc
 
Select * from Products order by 2

--Distinct 
 
Select distinct QuantityPerUnit from Products	

--Count

Select Count(*) from Products

Select Count(distinct QuantityPerUnit) from Products

--Where

Select * from Products where CategoryID = 1

--and

Select * from Products where UnitPrice >= 10.00 and UnitPrice <= 25.00 order by UnitPrice

-- between 

Select * from Products where UnitPrice between 10.00 and 25.00 order by UnitPrice

-- max

select Max(UnitPrice) from Products

--min

select Min(UnitPrice) from Products

-- alias

select Max(UnitPrice) as maxprice from Products

select Max(UnitPrice) as [max   price] from Products

-- count

Select Count(*) from Customers

Select Count(*) from Customers where Country = 'Germany'

--Sum

Select Sum(UnitsInStock) as  TotalStock from Products

-- In 

Select * from Customers where Country in ('Germany','France')

Select * from Products where CategoryID in (2,3,4)

Select Count(*) from Products where CategoryID in (2,3,4)

-- like

Select Count(*) from Products where ProductName like 'C%'
Select Count(*) from Products where ProductName like 'Cha%'

Select Count(*) from Products where ProductName like '%e'
Select Count(*) from Products where ProductName like '%ng'
Select Count(*) from Products where ProductName like '%es%'

Select Count(*) from Products where ProductName like 'C%e'
Select Count(*) from Products where ProductName like 'a_%'

Select Count(*) from Products where ProductName like '_a%'
Select Count(*) from Products where ProductName like '__a%'

-- join

Select * from products as p join Categories as c on p.CategoryID = c.CategoryID 
Select c.CategoryID,c.CategoryName,p.ProductID,p.ProductName,p.UnitPrice from products as p join Categories as c on p.CategoryID = c.CategoryID 





