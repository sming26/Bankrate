setwd('/Users/riyueyoutu/Desktop/USC/Courses/DSO522/project/')

#### --------------- read data ---------------
library(quantmod)
library(VIF)
## macro economy
# 10-Year Breakeven Inflation Rate, daily
getSymbols(Symbols = "T10YIE",src="FRED")
# 5-Year Forward Inflation Expectation Rate, daily
getSymbols(Symbols = "T5YIFR",src="FRED")
# 5-Year Breakeven Inflation Rate, daily
getSymbols(Symbols = "T5YIE",src="FRED")

## stock market
# Nasdaq indicators, daily
getSymbols("^IXIC", src = "yahoo")
# Dow Jones indicators, daily
getSymbols("^DJI", src = "yahoo")

## Federal Rate
# Effective Federal Funds Rate, weekly wed
getSymbols(Symbols = "FF",src="FRED")

## bonds
# CBOE 10-Year Treasury Note Volatility Futures, daily
getSymbols(Symbols = "VXTYN",src="FRED")
# 10-Year Treasury Constant Maturity Rate, daily
getSymbols(Symbols = "DGS10",src="FRED")
# 10-Year Treasury Inflation-indicatorsed Security, Constant Maturity, daily
getSymbols(Symbols = "DFII10",src="FRED")

## libor rate
# 1-month libor rate, daily
getSymbols(Symbols = 'USD1MTD156N', src = 'FRED')
# 3-month libor rate, daily
getSymbols(Symbols = 'USD3MTD156N', src = 'FRED')
# 6-month libor rate, daily
getSymbols(Symbols = 'USD6MTD156N', src = 'FRED')
# 12-month libor rate, daily
getSymbols(Symbols = 'USD12MD156N', src = 'FRED')

## employment
# Insured Unemployment Rate, weekly sat
getSymbols(Symbols = "IURSA",src="FRED")
# Continued Claims (Insured Unemployment), weekly sat
getSymbols(Symbols = "CCNSA",src="FRED")

## Housing
# mortgage 5/1 arm margin, weekly thur
getSymbols('MORTMRGN5US', src = 'FRED')
# Real Estate Loans, All Commercial Banks, weekly wed
getSymbols('RELACBW027NBOG', src = 'FRED')
# Real Estate Loans: Revolving Home Equity Loans, All Commercial Banks, weekly wed
getSymbols('RHEACBW027NBOG', src = 'FRED')
# Mortgage-backed securities held by the Federal Reserve: All Maturities, weekly wed
getSymbols('MBST', src = 'FRED')
# Origination Fees and Discount Points for 30-Year Fixed Rate Mortgage in the United State, weekly thur
getSymbols('MORTPTS30US', src = 'FRED')
# Origination Fees and Discount Points for 15-Year Fixed Rate Mortgage in the United State, weekly thur
getSymbols('MORTPTS15US', src = 'FRED')
# Origination Fees and Discount Points for 5/1-Year Adjustable Rate Mortgage in the United State, weekly thur
getSymbols('MORTPTS5US', src = 'FRED')

## target mortgage rates
# 30 years fixed, weekly thur
getSymbols('MORTGAGE30US', src = 'FRED')
# 15 years fixed, weekly thur
getSymbols('MORTGAGE15US', src = 'FRED') 
# 5/1 arm, weekly thur
getSymbols('MORTGAGE5US', src = 'FRED')

#### --------------- data preprocessing -----------------
# This function is to convert daily indicator to weekly and set cut-off at every thursday
day_to_week = function(table){
  new_name = paste(names(table), '_weekly', sep='')
  assign(new_name, apply.weekly(na.exclude(lag(table, 1)), mean), envir=.GlobalEnv)
}

# This function is to subset data based on time range
timeframe = function(table, start='2007-01-01', end='2017-11-17'){
  return(data.frame(table[index(table)>=start & index(table)<=end,], row.names = NULL))
}

## convert daily indicators to weekly
day_to_week(T10YIE)
day_to_week(T5YIFR)
day_to_week(T5YIE)
day_to_week(IXIC[,6])
day_to_week(DJI[,6])
day_to_week(VXTYN)
day_to_week(DGS10)
day_to_week(DFII10)
day_to_week(USD1MTD156N)
day_to_week(USD3MTD156N)
day_to_week(USD6MTD156N)
day_to_week(USD12MD156N)

# store all predictor names for future iteration use
ind_names = c('T10YIE_weekly', 'T5YIFR_weekly', 'T5YIE_weekly',
              'IXIC.Adjusted_weekly','DJI.Adjusted_weekly',
              'VXTYN_weekly', 'DGS10_weekly','DFII10_weekly', 
              'USD1MTD156N_weekly', 'USD3MTD156N_weekly', 'USD6MTD156N_weekly', 'USD12MD156N_weekly',
              'FF', 'IURSA', 'CCNSA', 'MORTMRGN5US', 'RELACBW027NBOG', 'RHEACBW027NBOG', 
              'MBST', 'MORTPTS30US', 'MORTPTS15US', 'MORTPTS5US')

## merge all indicators to one table
indicators07_17 = data.frame(date=index(MORTGAGE30US['2007-01-01/2017-11-16']))
for (i in ind_names){
  if (i %in% c('IURSA', 'CCNSA')) {
    # for the weekly indicators updated after Thursday, we need to take the lag-1 value
    indicators07_17 = cbind(indicators07_17, timeframe(eval(parse(text=i)), start='2006-12-30', end='2017-11-11'))
  } else {
    indicators07_17 = cbind(indicators07_17, timeframe(eval(parse(text=i))))  
  }
}

## merge three target mortgage rates to one table 
targets07_17 = data.frame(timeframe(MORTGAGE30US),
                          timeframe(MORTGAGE15US),
                          timeframe(MORTGAGE5US))

#### --------------- feature engineering -----------------
## create the lag-1 differences as the new features
xdif07_17 = indicators07_17
for (col in seq(ncol(xdif07_17))) {
  xdif07_17[,col] = xdif07_17[,col] - c(0, xdif07_17[-nrow(xdif07_17),col])
}
xdif07_17 = xdif07_17[-1,]
## convert the differences to the standardized value
for (col in seq(ncol(xdif07_17))) {
  xdif07_17[,col] = as.vector(prcomp(xdif07_17[,col])$x)
}

## create the lag-1 differences as the new targets
ydif07_17 = targets07_17
for (col in seq(ncol(ydif07_17))) {
  ydif07_17[,col] = ydif07_17[,col] - c(0, ydif07_17[-nrow(ydif07_17),col])
}
ydif07_17 = ydif07_17[-1,]
## convert the differences to the standardized value
for (col in seq(ncol(ydif07_17))) {
  ydif07_17[,col] = as.vector(prcomp(ydif07_17[,col])$x)
}

#### --------------- feature selection -----------------
# function to select significant features between X of different lags and Y with VIF
selectfeats = function(mort_index, weeklag, xtable, ytable) {
  nrows = nrow(ytable)
  feats = vif(ytable[(weeklag+1):nrows, mort_index], cbind(xtable[,-1], ytable)[1:(nrows-weeklag),], trace=F)$select
  return(feats)
}

## collect the significant features for 30-y mortgage from lag1 to lag12
feats30 = c()
for (i in seq(12)) {
  feats30 = c(feats30, selectfeats(1, i, xdif07_17, ydif07_17))
}
feats30 = colnames(cbind(xdif07_17[,-1], ydif07_17))[feats30[!duplicated(feats30)]]

## collect the significant features for 15-y mortgage from lag1 to lag12
feats15 = c()
for (i in seq(12)) {
  feats15 = c(feats15, selectfeats(2, i, xdif07_17, ydif07_17))
}
feats15 = colnames(cbind(xdif07_17[,-1], ydif07_17))[feats15[!duplicated(feats15)]]

## collect the significant features for 5/1 arm mortgage from lag1 to lag12
feats5 = c()
for (i in seq(12)) {
  feats5 = c(feats5, selectfeats(3, i, xdif07_17, ydif07_17))
}
feats5 = colnames(cbind(xdif07_17[,-1], ydif07_17))[feats5[!duplicated(feats5)]]

## generate full datasets including differenced values for 4 future weeks for each target respectively
mergeall = function(xtable, ytable, feats, mort_index) {
  xtable = cbind(xtable, ytable)[, c('date',feats)]
  for (col in seq(2, ncol(xtable))) {
    xtable[,col] = xtable[,col] - c(0, xtable[-nrow(xtable),col])
  }
  table = cbind(xtable, ytable[,mort_index])[-1,]
  for (i in seq(4)) {
    if (i == 1) {
      table[,paste0('week',i)] = c(table[2:nrow(table),ncol(table)], 0) - table[,ncol(table)]
    } else {
      table[,paste0('week',i)] = c(table[2:nrow(table),ncol(table)], 0)
    }
    table = table[-nrow(table),]
  }
  table = table[,-(ncol(table)-4)]
  return(table)
}

mort30 = mergeall(indicators07_17, targets07_17, feats30, 1)
mort15 = mergeall(indicators07_17, targets07_17, feats15, 2)
mort5 = mergeall(indicators07_17, targets07_17, feats5, 3)

write.csv(mort30, 'mort30.csv', row.names = F)
write.csv(mort15, 'mort15.csv', row.names = F)
write.csv(mort5, 'mort5.csv', row.names = F)
write.csv(targets07_17, 'morts.csv', row.names = F)
