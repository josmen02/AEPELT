/* CÓDIGO SAS */

/* Etiquetar variables */
data tfm.cierres_dia;
set tfm.cierres;
label
'Acerinox_SA'n = 'Acerinox'
'Actividades_de_Construccion_y_Se'n = 'ACS'
'Aena_SA'n = 'Aena'
'Alantra_Partners_SA'n = 'Alantra'
'Amadeus_IT_Group_SA'n = 'Amadeus'
'Banco_de_Sabadell_SA'n = 'Banco Sabadell'
'Banco_Santander_SA'n = 'Banco Santander'
'Bankinter_SA'n = 'Bankinter'
'Banco_Bilbao_Vizcaya_Argentaria'n = 'BBVA'
'CaixaBank_SA'n = 'CaixaBank'
'Cellnex_Telecom_SA'n = 'Cellnex Telecom'
'CIE_Automotive_SA'n = 'CIE Automotive'
'Enagas_SA'n = 'Enagas'
'Endesa_SA'n = 'Endesa'
'Ferrovial_SA'n = 'Ferrovial'
'Fluidra_SA'n = 'Fluidra'
'International_Consolidated_Airli'n = 'IAG'
'Iberdrola_SA'n = 'Iberdrola'
'Industria_de_Disenio_Textil_SA'n = 'Inditex'
'Indra_Sistemas_SA'n = 'Indra'
'Inmobiliaria_Colonial_SOCIMI_SA'n = 'Colonial'
'Mapfre_SA'n = 'Mapfre'
'Melia_Hotels_International_SA'n = 'Melia Hotels'
'Merlin_Properties_SOCIMI_SA'n = 'Merlin Properties'
'Naturgy_Energy_Group_SA'n = 'Naturgy'
'PharmaMar_SA'n = 'Pharma Mar'
'Repsol_SA'n = 'Repsol'
'Telefonica_SA'n = 'Telefonica'
'Viscofan_SA'n = 'Viscofan';
run;

/* Renombrar variables */
data tfm.visualizacion;
    set tfm.cierres_dia;
    rename Actividades_de_Construccion_y_Se = ACS;
    rename Banco_Bilbao_Vizcaya_Argentaria_ = BBVA;
    rename International_Consolidated_Airli = IAG;
    rename Industria_de_Disenio_Textil_SA = Inditex;
    rename Inmobiliaria_Colonial_SOCIMI_SA = Colonial;
    rename Melia_Hotels_International_SA = Melia_hotels;
run;

/* Gráfico conjunto de las variables */
proc sgplot data=tfm.visualizacion;
	title "Evolución Temporal de Precios de Cotizaciones";
	yaxis label="Precio";
    series x=Fecha y=Acciona;
    series x=Fecha y=Acerinox_SA;
    series x=Fecha y=ACS;
    series x=Fecha y=Aena_SA;
    series x=Fecha y=Alantra_Partners_SA;
    series x=Fecha y=Amadeus_IT_Group_SA;
    series x=Fecha y=Banco_de_Sabadell_SA;
    series x=Fecha y=Banco_Santander_SA;
    series x=Fecha y=Bankinter_SA;
    series x=Fecha y=BBVA;
    series x=Fecha y=CaixaBank_SA;
    series x=Fecha y=Cellnex_Telecom_SA;
    series x=Fecha y=CIE_Automotive_SA;
    series x=Fecha y=Enagas_SA;
    series x=Fecha y=Endesa_SA;
    series x=Fecha y=Ferrovial_SA;
    series x=Fecha y=Fluidra_SA;
    series x=Fecha y=IAG;
    series x=Fecha y=Iberdrola_SA;
    series x=Fecha y=Inditex;
    series x=Fecha y=Indra_Sistemas_SA;
    series x=Fecha y=Colonial;
    series x=Fecha y=Mapfre_SA;
    series x=Fecha y=Melia_hotels;
    series x=Fecha y=Merlin_Properties_SOCIMI_SA;
    series x=Fecha y=Naturgy_Energy_Group_SA;
    series x=Fecha y=PharmaMar_SA;
    series x=Fecha y=Repsol_SA;
    series x=Fecha y=Telefonica_SA;
    series x=Fecha y=Viscofan_SA;
run;

/* ANÁLISIS FACTORIAL */
proc factor data=tfm.cierres_dia corr plots=all outstat=est_fact out=project_fact
residuals nfact=5 msa rotate=quartimax plots=all ;
var acciona--Viscofan_SA;
pathdiagram fuzz=0.59 scale=0.9
DESIGNHEIGHT= 1200
DESIGNWIDTH= 1100;
run;

data tfm.factores;
set tfm.project_fact(keep=fecha FACTOR1 FACTOR2 FACTOR3 FACTOR4 FACTOR5);
rename FACTOR1=Bancario_y_grandes_empresas;
rename FACTOR2=Transporte_y_tecnologia;
rename FACTOR3=Industrial;
rename FACTOR4=Acciona;
rename FACTOR5=Energetico;
run;

/* MODELO ESM */

/*  Bancario_y_grandes_empresas   */
proc esm data=tfm.factores  outfor=tfm.PREDICCIONES_esm_1 back=14 lead=14 out=salida outfor=salida1 /*con IC de las pred y verla fiabilidad etc*/
plot=(acf pacf errors forecasts forecastsonly) print=(estimates forecasts  statistics);
 id fecha interval=day;
forecast Bancario_y_grandes_empresas / model=LINEAR;
run;
/*  Transporte_y_tecnologia   */
proc esm data=tfm.factores  outfor=tfm.PREDICCIONES_esm_2 back=14 lead=14 out=salida outfor=salida1 /*con IC de las pred y verla fiabilidad etc*/
plot=(acf pacf errors forecasts forecastsonly) print=(estimates forecasts  statistics);
 id fecha interval=day;
forecast Transporte_y_tecnologia / model=DAMPTREND;
run;
/*  Industrial   */
proc esm data=tfm.factores  outfor=tfm.PREDICCIONES_esm_3 back=14 lead=14 out=salida outfor=salida1 /*con IC de las pred y verla fiabilidad etc*/
plot=(acf pacf errors forecasts forecastsonly) print=(estimates forecasts  statistics);
 id fecha interval=day;
forecast Industrial / model=simple;
run;
/*  Acciona   */
proc esm data=tfm.factores  outfor=tfm.PREDICCIONES_esm_4 back=14 lead=14 out=salida outfor=salida1 /*con IC de las pred y verla fiabilidad etc*/
plot=(acf pacf errors forecasts forecastsonly) print=(estimates forecasts  statistics);
 id fecha interval=day;
forecast Acciona / model=double;
run;
/*  Energetico   */
proc esm data=tfm.factores  outfor=tfm.PREDICCIONES_esm_5 back=14 lead=14 out=salida outfor=salida1 /*con IC de las pred y verla fiabilidad etc*/
plot=(acf pacf errors forecasts forecastsonly) print=(estimates forecasts  statistics);
 id fecha interval=day;
forecast Energetico / model=double;
run;
/* Predicciones ESM */
data predicciones1;
set tfm.PREDICCIONES_esm_1(keep=fecha actual predict);
rename predict=FORECAST_FACT1
actual=Bancario_y_grandes_empresas;
run;
data predicciones2;
set tfm.PREDICCIONES_esm_2(keep=fecha actual predict);
rename predict=FORECAST_FACT2
actual=Transporte_y_tecnologia;
run;
data predicciones3;
set tfm.PREDICCIONES_esm_3(keep=fecha actual predict);
rename predict=FORECAST_FACT3
actual=Industrial;
run;
data predicciones4;
set tfm.PREDICCIONES_esm_4(keep=fecha actual predict);
rename predict=FORECAST_FACT4
actual=Acciona;
run;
data predicciones5;
set tfm.PREDICCIONES_esm_5(keep=fecha actual predict);
rename predict=FORECAST_FACT5
actual=Energetico;
run;
data tfm.predicciones_esm;
merge predicciones1 predicciones2 predicciones3 predicciones4 predicciones5;
by fecha;
run;

/* MODELO arima */

/* Separación train y test */
data tfm.factores_train;
set tfm.factores (firstobs=1111 obs=1476);
run;

data tfm.factores_test;
set tfm.factores (firstobs=1477 obs=1490);
run;

/*  Bancario_y_grandes_empresas   */
PROC ARIMA DATA= tfm.factores_train;
IDENTIFY VAR=Bancario_y_grandes_empresas NLAG=24;
outlier;
RUN;
PROC ARIMA DATA= tfm.factores_train ;
IDENTIFY VAR=Bancario_y_grandes_empresas(1) NLAG=24;
ESTIMATE noconstant q=(2) OUTEST=EST OUTMODEL=MODELO OUTSTAT=AJUSTE PLOT ;
FORECAST LEAD=14 ID=FECHA INTERVAL=day OUT=tfm.PREDICCIONES1 PRINTALL;
outlier;
RUN;
/*  Transporte_y_tecnologia   */
PROC ARIMA DATA= tfm.factores_train;
IDENTIFY VAR=Transporte_y_tecnologia NLAG=24;
outlier;
RUN;
PROC ARIMA DATA= tfm.factores_train;
IDENTIFY VAR=Transporte_y_tecnologia(1) NLAG=24;
ESTIMATE  noconstant OUTEST=EST OUTMODEL=MODELO OUTSTAT=AJUSTE PLOT ;
FORECAST LEAD=14 ID=FECHA INTERVAL=day OUT=tfm.PREDICCIONES2 PRINTALL;
outlier;
RUN;
/*  Industrial   */
PROC ARIMA DATA= tfm.factores_train;
IDENTIFY VAR=Industrial NLAG=24;
outlier;
RUN;
PROC ARIMA DATA= tfm.factores_train;
IDENTIFY VAR=Industrial(1) NLAG=24;
ESTIMATE noconstant  OUTEST=EST OUTMODEL=MODELO OUTSTAT=AJUSTE PLOT ;
FORECAST LEAD=14 ID=FECHA INTERVAL=day OUT=tfm.PREDICCIONES3 PRINTALL;
outlier;
RUN;
/*  Acciona   */
PROC ARIMA DATA= tfm.factores_train;
IDENTIFY VAR=Acciona NLAG=24;
outlier;
RUN;
PROC ARIMA DATA= tfm.factores_train;
IDENTIFY VAR=Acciona(1) NLAG=24;
ESTIMATE noconstant  OUTEST=EST OUTMODEL=MODELO OUTSTAT=AJUSTE PLOT ;
FORECAST LEAD=14 ID=FECHA INTERVAL=day OUT=tfm.PREDICCIONES4 PRINTALL;
outlier;
RUN;
/*  Energetico   */
PROC ARIMA DATA= tfm.factores_train;
IDENTIFY VAR=Energetico NLAG=24;
outlier;
RUN;
PROC ARIMA DATA= tfm.factores_train;
IDENTIFY VAR=Energetico(1) NLAG=24;
ESTIMATE noconstant  OUTEST=EST OUTMODEL=MODELO OUTSTAT=AJUSTE PLOT ;
FORECAST LEAD=14 ID=FECHA INTERVAL=day OUT=tfm.PREDICCIONES5 PRINTALL;
outlier;
RUN;

/* Predicciones ARIMA */
data predicciones1;
set tfm.predicciones1(keep=fecha Bancario_y_grandes_empresas forecast);
rename FORECAST=FORECAST_FACT1;
run;
data predicciones2;
set tfm.predicciones2(keep=fecha Transporte_y_tecnologia forecast);
rename FORECAST=FORECAST_FACT2;
run;
data predicciones3;
set tfm.predicciones3(keep=fecha Industrial forecast);
rename FORECAST=FORECAST_FACT3;
run;
data predicciones4;
set tfm.predicciones4(keep=fecha Acciona forecast);
rename FORECAST=FORECAST_FACT4;
run;
data predicciones5;
set tfm.predicciones5(keep=fecha Energetico forecast);
rename FORECAST=FORECAST_FACT5;
run;
data tfm.predicciones;
merge predicciones1 predicciones2 predicciones3 predicciones4 predicciones5;
by fecha;
run;


/* CÁLCULO DE LAS PREDICCIONES PARA LAS COTIZACIONES ORIGINALES */

data tfm.factores_pred_esm;
set tfm.predicciones_esm (keep=FORECAST_FACT1-FORECAST_FACT5);
run;
data tfm.nombres_fila_esm;
set tfm.predicciones_esm(keep=fecha);run;
data tfm.cargas;
set tfm.cargas (keep=factor1-factor5);
run;
proc print data=tfm.cargas; run;
/* Transponer los datos */
proc transpose data=tfm.factores_pred_esm out=tfm.factores_pred_esm_t;
    var _all_; /* Transponer todas las variables */
run;
data tfm.factores_pred_esm_t;
set tfm.factores_pred_esm_t (keep=col1-col1490);
run;
/* Mostrar los datos transpuestos */
proc print data=tfm.factores_pred_esm_t; run;
proc iml;
use tfm.cargas;
read all var _NUM_ into A;
close tfm.cargas;

use tfm.factores_pred_esm_t;
read all var _NUM_ into B;
close tfm.factores_pred_esm_t;

/* Multiplicar matrices */
C = A * B;
create tfm.matriz_C_esm from C;
append from C;
close tfm.matriz_C_esm;
print C;
quit;
proc transpose data=tfm.matriz_C_esm out=tfm.matriz_C_esm_t;
    var _all_ ; /* Transponer todas las variables */
run;
/* Matriz de predicciones estandarizadas */
data tfm.matriz_C_esm_t;
set tfm.matriz_C_esm_t (keep=col1-col30);
run;

/* Renombrar las columnas */
data tfm.cierres_dia_pred_est_esm;
set tfm.nombres_fila_esm;
set tfm.matriz_C_esm_t;
rename col1 = Acciona 
       col2 = Acerinox_SA 
       col3 = ACS 
       col4 = Aena_SA 
       col5 = Alantra_Partners_SA 
       col6 = Amadeus_IT_Group_SA 
       col7 = Banco_de_Sabadell_SA 
       col8 = Banco_Santander_SA 
       col9 = Bankinter_SA 
       col10 = BBVA
       col11 = CaixaBank_SA 
       col12 = Cellnex_Telecom_SA 
       col13 = CIE_Automotive_SA 
       col14 = Enagas_SA 
       col15 = Endesa_SA 
       col16 = Ferrovial_SA 
       col17 = Fluidra_SA 
       col18 = IAG 
       col19 = Iberdrola_SA 
       col20 = INDITEX 
       col21 = Indra_Sistemas_SA 
       col22 = Colonial
       col23 = Mapfre_SA 
       col24 = Melia_Hotels
       col25 = Merlin_Properties_SOCIMI_SA 
       col26 = Naturgy_Energy_Group_SA 
       col27 = PharmaMar_SA 
       col28 = Repsol_SA 
       col29 = Telefonica_SA 
       col30 = Viscofan_SA;
run;
/* Calcular medias y desviaciones típicas*/
proc means data=tfm.cierres_dia ;run;

/* Desestandarizar las predicciones*/
data tfm.normalizar_esm;
set tfm.cierres_dia_pred_est_esm;
mean_Acciona = 141.4782887;
mean_Acerinox_SA = 9.5138879;
mean_ACS = 25.9976436;
mean_Aena_SA = 136.9523894;
mean_Alantra_Partners_SA = 10.7464497;
mean_Amadeus_IT_Group_SA = 57.0272483;
mean_Banco_de_Sabadell_SA = 0.7201764;
mean_Banco_Santander_SA = 2.9127164;
mean_Bankinter_SA = 4.7808973;
mean_BBVA = 5.1822091;
mean_CaixaBank_SA = 2.8852718;
mean_Cellnex_Telecom_SA = 41.9383419;
mean_CIE_Automotive_SA = 22.7371812;
mean_Enagas_SA = 18.8207181;
mean_Endesa_SA = 20.2972550;
mean_Ferrovial_SA = 25.5050940;
mean_Fluidra_SA = 20.4584966;
mean_IAG = 1.8649764;
mean_Iberdrola_SA = 10.5764772;
mean_INDITEX = 27.8730336;
mean_Indra_Sistemas_SA = 9.5581242;
mean_Colonial = 7.3766309;
mean_Mapfre_SA = 1.7896393;
mean_Melia_Hotels = 5.7342792;
mean_Merlin_Properties_SOCIMI_SA = 8.8796409;
mean_Naturgy_Energy_Group_SA = 23.4877100;
mean_PharmaMar_SA = 65.2652405;
mean_Repsol_SA = 11.4464718;
mean_Telefonica_SA = 3.9884493;
mean_Viscofan_SA = 57.3611611;
std_Acciona = 33.1110877;
std_Acerinox_SA = 1.4813188;
std_ACS = 4.7878327;
std_Aena_SA = 14.7257438;
std_Alantra_Partners_SA = 1.8960682;
std_Amadeus_IT_Group_SA = 7.2478190;
std_Banco_de_Sabadell_SA = 0.2864304;
std_Banco_Santander_SA = 0.5827466;
std_Bankinter_SA = 1.2049391;
std_BBVA = 1.5540608;
std_CaixaBank_SA = 0.7142545;
std_Cellnex_Telecom_SA = 7.3532968;
std_CIE_Automotive_SA = 4.0726435;
std_Enagas_SA = 2.1358195;
std_Endesa_SA = 2.1881994;
std_Ferrovial_SA = 3.0963762;
std_Fluidra_SA = 7.6889905;
std_IAG = 0.6897961;
std_Iberdrola_SA = 0.7837876;
std_INDITEX = 4.7678480;
std_Indra_Sistemas_SA = 2.3456946;
std_Colonial = 1.6053474;
std_Mapfre_SA = 0.1963109;
std_Melia_Hotels = 1.1493956;
std_Merlin_Properties_SOCIMI_SA = 1.3093287;
std_Naturgy_Energy_Group_SA = 4.0712297;
std_PharmaMar_SA = 22.9040623;
std_Repsol_SA = 2.6539305;
std_Telefonica_SA = 0.6003938;
std_Viscofan_SA = 4.0943087;
Acciona_pred = (std_Acciona * Acciona) + mean_Acciona;
Acerinox_SA_pred = (std_Acerinox_SA * Acerinox_SA) + mean_Acerinox_SA;
ACS_pred = (std_ACS * ACS) + mean_ACS;
Aena_SA_pred = (std_Aena_SA * Aena_SA) + mean_Aena_SA;
Alantra_Partners_SA_pred = (std_Alantra_Partners_SA * Alantra_Partners_SA) + mean_Alantra_Partners_SA;
Amadeus_IT_Group_SA_pred = (std_Amadeus_IT_Group_SA * Amadeus_IT_Group_SA) + mean_Amadeus_IT_Group_SA;
Banco_de_Sabadell_SA_pred = (std_Banco_de_Sabadell_SA * Banco_de_Sabadell_SA) + mean_Banco_de_Sabadell_SA;
Banco_Santander_SA_pred = (std_Banco_Santander_SA * Banco_Santander_SA) + mean_Banco_Santander_SA;
Bankinter_SA_pred = (std_Bankinter_SA * Bankinter_SA) + mean_Bankinter_SA;
BBVA_pred = (std_BBVA * BBVA) + mean_BBVA;
CaixaBank_SA_pred = (std_CaixaBank_SA * CaixaBank_SA) + mean_CaixaBank_SA;
Cellnex_Telecom_SA_pred = (std_Cellnex_Telecom_SA * Cellnex_Telecom_SA) + mean_Cellnex_Telecom_SA;
CIE_Automotive_SA_pred = (std_CIE_Automotive_SA * CIE_Automotive_SA) + mean_CIE_Automotive_SA;
Enagas_SA_pred = (std_Enagas_SA * Enagas_SA) + mean_Enagas_SA;
Endesa_SA_pred = (std_Endesa_SA * Endesa_SA) + mean_Endesa_SA;
Ferrovial_SA_pred = (std_Ferrovial_SA * Ferrovial_SA) + mean_Ferrovial_SA;
Fluidra_SA_pred = (std_Fluidra_SA * Fluidra_SA) + mean_Fluidra_SA;
IAG_pred = (std_IAG * IAG) + mean_IAG;
Iberdrola_SA_pred = (std_Iberdrola_SA * Iberdrola_SA) + mean_Iberdrola_SA;
INDITEX_pred = (std_INDITEX * INDITEX) + mean_INDITEX;
Indra_Sistemas_SA_pred = (std_Indra_Sistemas_SA * Indra_Sistemas_SA) + mean_Indra_Sistemas_SA;
Colonial_pred = (std_Colonial * Colonial) + mean_Colonial;
Mapfre_SA_pred = (std_Mapfre_SA * Mapfre_SA) + mean_Mapfre_SA;
Melia_Hotels_pred = (std_Melia_Hotels * Melia_Hotels) + mean_Melia_Hotels;
Merlin_Properties_SOCIMI_SA_pred = (std_Merlin_Properties_SOCIMI_SA * Merlin_Properties_SOCIMI_SA) + mean_Merlin_Properties_SOCIMI_SA;
Naturgy_Energy_Group_SA_pred = (std_Naturgy_Energy_Group_SA * Naturgy_Energy_Group_SA) + mean_Naturgy_Energy_Group_SA;
PharmaMar_SA_pred = (std_PharmaMar_SA * PharmaMar_SA) + mean_PharmaMar_SA;
Repsol_SA_pred = (std_Repsol_SA * Repsol_SA) + mean_Repsol_SA;
Telefonica_SA_pred = (std_Telefonica_SA * Telefonica_SA) + mean_Telefonica_SA;
Viscofan_SA_pred = (std_Viscofan_SA * Viscofan_SA) + mean_Viscofan_SA;
run;
/* Seleccionar solo las variables calculadas*/
data tfm.cierres_dia_pred_esm;
set tfm.normalizar_esm (keep= fecha Acciona_pred Acerinox_SA_pred ACS_pred Aena_SA_pred Alantra_Partners_SA_pred 
Amadeus_IT_Group_SA_pred Banco_de_Sabadell_SA_pred Banco_Santander_SA_pred Bankinter_SA_pred BBVA_pred
CaixaBank_SA_pred Cellnex_Telecom_SA_pred CIE_Automotive_SA_pred Enagas_SA_pred Endesa_SA_pred 
Ferrovial_SA_pred Fluidra_SA_pred IAG_pred Iberdrola_SA_pred INDITEX_pred Indra_Sistemas_SA_pred 
Colonial_pred Mapfre_SA_pred Melia_Hotels_pred Merlin_Properties_SOCIMI_SA_pred 
Naturgy_Energy_Group_SA_pred PharmaMar_SA_pred Repsol_SA_pred Telefonica_SA_pred Viscofan_SA_pred)
;
proc sort data=tfm.cierres_dia;
by fecha;
run;

proc sort data=tfm.cierres_dia_pred_esm;
by fecha;
run;
/* Crear conjunto de datos con las variables originales y sus respectivas predicciones*/
data tfm.comparacion_esm;
merge  tfm.cierres_dia tfm.cierres_dia_pred_esm;
by fecha;
rename Actividades_de_Construccion_y_Se=ACS;
rename Banco_Bilbao_Vizcaya_Argentaria_=BBVA;
rename International_Consolidated_Airli=IAG;
rename Industria_de_Disenio_Textil_SA=Inditex;
rename Inmobiliaria_Colonial_SOCIMI_SA=Colonial;
rename Melia_Hotels_International_SA=Melia_hotels;
run;

/* Lista de empresas y sus respectivas predicciones */
%let empresas = Acciona Acerinox_SA ACS Aena_SA Alantra_Partners_SA Amadeus_IT_Group_SA Banco_de_Sabadell_SA Banco_Santander_SA Bankinter_SA BBVA CaixaBank_SA Cellnex_Telecom_SA CIE_Automotive_SA Enagas_SA Endesa_SA Ferrovial_SA Fluidra_SA IAG Iberdrola_SA INDITEX Indra_Sistemas_SA Colonial Mapfre_SA Melia_Hotels Merlin_Properties_SOCIMI_SA Naturgy_Energy_Group_SA PharmaMar_SA Repsol_SA Telefonica_SA Viscofan_SA;
%let titulo = Cotización Real y Predicción para &empresa.;
/* Bucle para generar los gráficos de las predicciones y valores reales */
%macro generar_graficos;
    %do i = 1 %to %sysfunc(countw(&empresas.));
        %let empresa = %scan(&empresas., &i.);
        
        proc sgplot data=tfm.comparacion_esm;
            series x=FECHA y=&empresa. / legendlabel="&empresa." lineattrs=(color=blue);
            series x=FECHA y=&empresa._pred / legendlabel="&empresa. Predicción" lineattrs=(color=red);
			title "&titulo."; /* Cambiar el título del gráfico */
        run;
    %end;
%mend generar_graficos;

/* Ejecutar el macro */
%generar_graficos;
