
-- IMPORTANT: make sure that on the left hand side of the execute buttom above, the tfg-inso-db is selected


CREATE USER [dataFactoryTFGinso] FROM EXTERNAL PROVIDER;

ALTER ROLE db_owner ADD MEMBER [dataFactoryTFGinso];