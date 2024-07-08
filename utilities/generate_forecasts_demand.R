# START ----
.Packages <- 'tictoc, tidyverse, readxl, tseries, forecast'; {
    .Packages <- strsplit(.Packages, ', ')[[1]]
    
    .curScript <- rstudioapi::getSourceEditorContext()$path
    
    .inactive <- function() {
        if (exists('.active'))
            if (.active == .curScript)
                return (FALSE)
        return (TRUE)
    }
    
    .include <- function(p) {
        if (!is.element(p, rownames(installed.packages())))
            install.packages(p, quiet=TRUE)
        .nOldPackages <- length(.packages())
        suppressPackageStartupMessages(require(p, quietly=TRUE, character.only=TRUE))
        return (.packages()[1:(length(.packages()) - .nOldPackages)])
    }
    
    .exclude <- function(packages)
        lapply(paste0('package:', packages), function(p)
            suppressWarnings(detach(p, character.only=TRUE, unload=TRUE)))
    
    if (.inactive()) {
        .prvDirectory <- getwd()
        if (exists('.allPackages')) {
            if (length(.prvPackages <- names(.allPackages)))
                .exclude(unlist(.allPackages))
        } else if (length(.packages()) > 7) {
            .exclude(.prvPackages <- .packages()[1:(length(.packages()) - 7)])
        } else
            .prvPackages <- c()
        .prvOs <- setdiff(objects(all.names=TRUE), c('.Packages', '.curScript', '.inactive', '.include', '.exclude'))
        save(list=.prvOs, file='~/R/.prvEnvironment.RData', envir=.GlobalEnv)
        rm(list=.prvOs)
        .active <- .curScript
        .allPackages <- sapply(.Packages, .include, simplify=FALSE)
    }
    
    .curDirectory <- ''
    
    if (.curDirectory == '') 
        .curDirectory <- dirname(.curScript)
    
    setwd(ifelse(.curDirectory == '', '~', .curDirectory))
    
    .oldPackages <- setdiff(names(.allPackages), .Packages)
    
    for (p in .oldPackages) {
        .exclude(.allPackages[[p]])
        .allPackages[[p]] <- NULL
    }
    
    .newPackages <- setdiff(.Packages, names(.allPackages))
    
    for (p in .newPackages)
        .allPackages[[p]] <- .include(p)
    
    rm(p)
    cat('\nCurrent File: ', ifelse(.active!='', .active, 'unsaved'), '\n\n', sep='')
}




# EXECUTION ----
data <- read_excel("./data/USWeeklyProductSupplied.xlsx", sheet="Data 1", col_names=c("Date", "n_barrels_total", "n_barrels_motorgasoline"), range="A17:C1755") %>% select(Date, n_barrels_motorgasoline)
data_ts <- ts(data$n_barrels_motorgasoline, start=c(1991, 6), frequency=52)
arima_model <- readRDS("./models/model_demand.rds")
forecast_demand <- forecast(arima_model, h=3944)
probabilities_events <- read.csv("./data/probabilities_events.csv", row.names=1)
probabilities_events_weekly <- data.frame(matrix(numeric(), nrow=nrow(probabilities_events), ncol=52), row.names=rownames(probabilities_events))
colnames(probabilities_events_weekly) <- paste0("week", 1:52)
distributions_impact <- list(
    normal = c(1.0, 0.0),  # No impact for normal weeks
    hurricanes = c(0.995, 0.05),
    earthquakes = c(0.99, 0.03),
    recession = c(0.99, 0.05),
    economic_boom = c(1.01, 0.05),
    oil_supply_disruptions = c(0.985, 0.07),
    peace_agreements = c(1.005, 0.02),
    renewable_energy_breakthrough = c(0.98, 0.1),
    improved_fuel_efficiency = c(0.995, 0.02),
    new_regulations = c(0.99, 0.03),
    subsidies_for_petroleum_products = c(1.01, 0.04),
    pandemic = c(0.965, 0.1),
    large_scale_events = c(0.975, 0.02),
    severe_winter_weather = c(1.002, 0.015),
    summer_travel_season = c(1.001, 0.05),
    price_speculation = c(1.001, 0.03)
)
weeks <- c(29:52, rep(1:52, 2099 - 2024), 1:20)
events <- rownames(probabilities_events)

for (w in 1:52) {
    probabilities_events_weekly[w] <- probabilities_events[ceiling(12*w/52)]
}


estimate_standard_deviation <- function(lower, upper) {
    #              MOE       /  Z
    return ((upper - lower)/2/4.128)
}


sample_impact <- function(event) {
    mean <- distributions_impact[[event]][1]
    sd <- distributions_impact[[event]][2]
    return(rnorm(1, mean, sd))
}


.Main <- function() {
    for (i in 0:10) {
        seed <- i
        set.seed(seed)
        
        data_ts_withshock <- data_ts
        arima_model_withshock <- arima_model
        forecast_demand_withshock <- forecast_demand
        forecast_demand_mean <- data.frame(mean=numeric(length=3944), value=numeric(length=3944))
        forecast_demand_upper <- matrix(numeric(length=3944), nrow=3944, ncol=2)
        forecast_demand_lower <- matrix(numeric(length=3944), nrow=3944, ncol=2)
        forecast_events <- numeric(length=3944)
        
        for (i in 1:3944) {
            # Forecast the demand for the next week
            next_week_demand <- forecast(arima_model_withshock, h=1)
            
            # Calculate the standard deviation
            # standard_deviation <- (next_week_demand$mean - next_week_demand$lower[2])/2
            standard_deviation <- estimate_standard_deviation(next_week_demand$lower[2], next_week_demand$upper[2])
            
            # Calculate the deviation from the mean
            deviation <- rnorm(1, 0, standard_deviation)
            # deviation <- 0
            
            # Calculate the impact of a random shock event on the demand
            i_event <- sample(1:length(rownames(probabilities_events)), 1, prob=probabilities_events_weekly[[weeks[i]]])
            impact <- sample_impact(events[i_event])
            forecast_events[i] <- i_event
            
            # Generate a random demand value
            forecast_demand_mean[i,] <- c(floor(next_week_demand$mean), floor(impact*(next_week_demand$mean + deviation)))
            forecast_demand_upper[i,] <- impact*next_week_demand$upper[1,]
            forecast_demand_lower[i,] <- impact*next_week_demand$lower[1,]
            
            # Add the new demand value to the time series
            data_ts_withshock <- ts(c(data_ts_withshock, forecast_demand_mean$value[i]), start=c(1991, 6), frequency=52)
            
            # Fit an ARIMA model to the updated time series
            arima_model_withshock <- arima(
                data_ts_withshock, 
                order=c(1, 1, 2),
                include.mean=TRUE
            )
        }
        
        forecast_demand_withshock$mean <- ts(forecast_demand_mean$value, start=c(2024, 29), frequency=52)
        forecast_demand_withshock$upper <- ts(forecast_demand_upper, start=c(2024, 29), frequency=52)
        forecast_demand_withshock$lower <- ts(forecast_demand_lower, start=c(2024, 29), frequency=52)
        forecast_events <- ts(forecast_events, start=c(2024, 29), frequency=52)
        
        plot(forecast_demand_withshock, main="Forecasted Demand of Petroleum Products", xlab="Weeks", ylab="Demand (in 1000 barrels)")
        
        write.csv(
            cbind(Event=forecast_events, Mean=forecast_demand_mean$mean, data.frame(forecast_demand_withshock)),
            file=paste0("./forecasts/test", seed, ".csv")
        )
    }
    
    return ()
}


if (!sys.nframe() | sys.nframe() == 4) {
    tic()
    
    if (!is.null(.answer <- .Main())) 
        print(.answer)
    
    toc()
}




# END ----
if (0) {
    if (!exists('.Packages')) {  # create default sessions
        if (length(.packages()) > 7)
            lapply(paste0('package:', .packages()[1:(length(.packages()) - 7)]), function(p) 
                suppressWarnings(detach(p, character.only=TRUE, unload=TRUE)))
        rm(list=objects(all.names=TRUE))
        setwd('~')
    }
    
    else{  # restore previous script
        if (length(.allPackages))
            lapply(paste0('package:', unlist(.allPackages)), function(p) 
                suppressWarnings(detach(p, character.only=TRUE, unload=TRUE)))
        rm(list=objects(all.names=TRUE))
        load('~/R/.prvEnvironment.RData')
        invisible(lapply(.prvPackages, function(p) 
            suppressPackageStartupMessages(require(p, quietly=TRUE, character.only=TRUE))))
        setwd(.prvDirectory)
        rm(.prvPackages, .prvDirectory)
    }
    
    cat('\n', ifelse(exists('.active'), paste('Current File:', .active), 'R session reset...'), '\n\n', sep='')
}
