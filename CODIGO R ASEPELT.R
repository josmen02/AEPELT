# Librería para extraer los datos 
library(quantmod)

# Símbolos de las cotizaciones
simbolos <- c("ANA.MC","ACX.MC", "ACS.MC", "AENA.MC", "ALM.MC", "AMS.MC", 
              "SAB.MC", "SAN.MC", "BKT.MC", "BBVA.MC", 
              "CABK.MC", "CLNX.MC", "CIE.MC", "ENG.MC", 
              "ELE.MC", "FER.MC", "FDR.MC", "IAG.MC", 
              "IBE.MC", "ITX.MC", "IDR.MC", "COL.MC", "MAP.MC", 
              "MEL.MC", "MRL.MC", "NTGY.MC", "PHM.MC", "REP.MC", 
              "TEF.MC", "VIS.MC")

# Crear una lista para almacenar los datos de cierre de cotización
lista_cierres <- list()

# Función que extrae las cotizaciones
for (simbolo in simbolos) {
  datos <- getSymbols(simbolo, from = "2020-01-01", to = "2024-01-31", src = "yahoo", auto.assign = FALSE)
  lista_cierres[[simbolo]]<-datos[,4]
}
lista_cierres

# Nombres de las empresas
nombres_empresas <- c("Acciona", "Acerinox_SA", "Actividades_de_Construccion_y_Servicios_SA", "Aena_SA",
                      "Alantra_Partners_SA", "Amadeus_IT_Group_SA", 
                      "Banco_de_Sabadell_SA", "Banco_Santander_SA", "Bankinter_SA", 
                      "Banco_Bilbao_Vizcaya_Argentaria_SA", "CaixaBank_SA", 
                      "Cellnex_Telecom_SA", "CIE_Automotive_SA", 
                      "Enagas_SA", "Endesa_SA", "Ferrovial_SA", "Fluidra_SA", 
                      "International_Consolidated_Airlines_Group_SA", "Iberdrola_SA", 
                      "Industria_de_Disenio_Textil_SA", "Indra_Sistemas_SA", "Inmobiliaria_Colonial_SOCIMI_SA", 
                      "Mapfre_SA", "Melia_Hotels_International_SA", "Merlin_Properties_SOCIMI_SA", 
                      "Naturgy_Energy_Group_SA", "PharmaMar_SA", "Repsol_SA", 
                      "Telefonica_SA", "Viscofan_SA")

# Asignar nombres a los símbolos
lista_cierres_empresas<-lista_cierres


cierres_empresas <- as.data.frame(lista_cierres_empresas)
names(cierres_empresas) <- nombres_empresas

row.names(cierres_empresas)
cierres_empresas
# Definir la ruta completa al archivo CSV
ruta_archivo <- "C:/Users/jose antonio mendoza/Documents/UNI/MASTER/TFM/cierres_empresas.csv"

# Escribir el data frame en un archivo CSV con el separador ;
write.csv(cierres_empresas, ruta_archivo, row.names = TRUE, fileEncoding = "UTF-8")




# Iterar sobre cada serie temporal en la lista y convertirla a serie mensual
lista_series_mensuales <- lapply(lista_cierres, function(x) {
  # Convertir la serie temporal a una serie mensual
  serie_mensual <- to.monthly(x)
  
  # Convertir el índice de vuelta a tipo Date
  index(serie_mensual) <- as.Date(index(serie_mensual))
  
  return(serie_mensual)
})

lista_series_mensuales

# Definir una función para calcular la media entre x.Open y x.Close en una serie temporal mensual
calcular_media_open_close <- function(serie_temporal) {
  serie_temporal$x.Mean <- (serie_temporal$x.Open + serie_temporal$x.Close) / 2
  return(serie_temporal)
}

# Aplicar la función a cada serie temporal en la lista
lista_series_con_media <- lapply(lista_series_mensuales, calcular_media_open_close)

lista_series_con_media


head(lista_cierres)


