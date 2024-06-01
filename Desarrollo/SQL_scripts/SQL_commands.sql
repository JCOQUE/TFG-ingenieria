DROP TABLE diarioDB;
GO

CREATE SCHEMA [diarioDB];
GO


-- IMPORTANT: make sure that on the left hand side of the execute buttom above, the tfg-inso-db is selected

CREATE TABLE diarioDB (
    ID int NOT NULL, -- Movimiento column needs to be deleted. It was giving problems
    Fecha DATETIME, 
    Cuenta VARCHAR(100), 
    NoCuenta VARCHAR(50),
    Debe DECIMAL(18,5), -- It keeps the first 5 decimals. It needs 5 so that the balance in Debe and Haber in the Dashboard is correct
    Haber DECIMAL(18,5), -- It keeps the first 5 decimals. It needs 5 so that the balance in Debe and Haber in the Dashboard is correct
    Entidad VARCHAR(100),
    Compras DECIMAL(18,2),
    Ventas DECIMAL(18,2),
    NoGrupo VARCHAR(50),
	type VARCHAR(20)
);
GO

select * 
from diarioDB

SELECT max(ID) AS max_id FROM diarioDB;


SELECT TOP 1 * FROM diarioDB ORDER BY Fecha DESC;

