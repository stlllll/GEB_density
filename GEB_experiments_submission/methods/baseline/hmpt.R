# Shared preprocessing, scoring, and checkpoint logic for HMPT.

dataset_names <- c(
  "gaussian_mixture_low_d",
  "gaussian_mixture_high_d_sparse",
  "hyperrectangle_structure",
  "dry_bean", "isolet", "household_power"
)
dataset_transforms <- list(
  gaussian_mixture_low_d=c("mixture", "normal"),
  gaussian_mixture_high_d_sparse=c("mixture", "normal"),
  hyperrectangle_structure=c("mixture", "normal"),
  dry_bean="normal",
  isolet="normal",
  household_power="mixture"
)
n_repetitions <- 20L
cdf_eps <- 1e-12
uniform_weight <- 0.999
normal_weight <- 0.001
expected_dimensions <- c(
  gaussian_mixture_low_d=2L,
  gaussian_mixture_high_d_sparse=50L,
  hyperrectangle_structure=5L,
  dry_bean=16L,
  isolet=100L,
  household_power=6L
)

split_path <- function(project_root, dataset, repetition, split) {
  file.path(project_root, "data", "train_test", dataset,
            sprintf("repetition_%02d_%s.npz", repetition, split))
}

result_path <- function(project_root, dataset) {
  file.path(project_root, "results", "density_estimation",
            paste0(dataset, "_results.csv"))
}

read_split <- function(project_root, dataset, repetition, split, np) {
  path <- split_path(project_root, dataset, repetition, split)
  if (!file.exists(path)) stop("Missing split: ", path)
  archive <- np$load(normalizePath(path, winslash="/", mustWork=TRUE),
                     allow_pickle=FALSE)
  on.exit(archive$close(), add=TRUE)
  x <- reticulate::py_to_r(archive[["x"]])
  expected <- expected_dimensions[[dataset]]
  if (!is.null(expected) && ncol(x) != expected) {
    stop("Stale ", dataset, " split has D=", ncol(x), ", expected D=",
         expected, ". Regenerate it with --reset-results.")
  }
  x
}

fit_tree_transform <- function(x_train, method) {
  mean <- colMeans(x_train)
  centered <- sweep(x_train, 2, mean, "-")
  sd <- sqrt(colMeans(centered^2))
  if (any(!is.finite(sd)) || any(sd <= .Machine$double.eps)) {
    stop("Degenerate training coordinate")
  }
  if (method == "normal") return(list(method=method, mean=mean, sd=sd))
  if (method != "mixture") stop("Unknown transform: ", method)
  lower <- apply(x_train, 2, min)
  width <- apply(x_train, 2, max) - lower
  if (any(!is.finite(width)) || any(width <= .Machine$double.eps)) {
    stop("Degenerate training range")
  }
  list(method=method, lower=lower, width=width, mean=mean, sd=sd)
}

apply_tree_transform <- function(x, parameters, jacobian=FALSE) {
  q <- sweep(sweep(x, 2, parameters$mean, "-"), 2, parameters$sd, "/")
  if (parameters$method == "normal") {
    u <- pmin(pmax(pnorm(q), cdf_eps), 1-cdf_eps)
    if (!jacobian) return(u)
    log_jacobian <- rowSums(dnorm(q, log=TRUE)) - sum(log(parameters$sd))
    return(list(u=u, log_jacobian=log_jacobian))
  }
  uniform_cdf <- pmin(pmax(
    sweep(sweep(x, 2, parameters$lower, "-"), 2, parameters$width, "/"),
    0), 1)
  u <- uniform_weight*uniform_cdf + normal_weight*pnorm(q)
  u <- pmin(pmax(u, cdf_eps), 1-cdf_eps)
  if (!jacobian) return(u)
  inside <- sweep(x, 2, parameters$lower, ">=") &
            sweep(x, 2, parameters$lower+parameters$width, "<=")
  uniform_density <- sweep(inside*uniform_weight, 2, parameters$width, "/")
  normal_density <- sweep(dnorm(q)*normal_weight, 2, parameters$sd, "/")
  log_jacobian <- rowSums(log(uniform_density+normal_density))
  list(u=u, log_jacobian=log_jacobian)
}

numeric_result_rows <- function(table) {
  repetitions <- suppressWarnings(as.integer(table$repetition))
  table <- table[!is.na(repetitions), , drop=FALSE]
  table$repetition <- repetitions[!is.na(repetitions)]
  table
}

add_summary_rows <- function(table) {
  table <- numeric_result_rows(table)
  table <- table[order(table$repetition), , drop=FALSE]
  mean_row <- as.list(rep(NA, ncol(table)))
  se_row <- as.list(rep(NA, ncol(table)))
  names(mean_row) <- names(se_row) <- names(table)
  mean_row$repetition <- "Mean"
  se_row$repetition <- "Standard error"
  for (column in setdiff(names(table), c("repetition", "n", "D"))) {
    values <- suppressWarnings(as.numeric(table[[column]]))
    if (grepl("^GEB_.*_running_time$", column)) {
      values <- values[table$repetition != 1L]
    }
    values <- values[is.finite(values)]
    if (length(values)) mean_row[[column]] <- mean(values)
    if (length(values) > 1L) se_row[[column]] <- sd(values) / sqrt(length(values))
  }
  rbind(table, as.data.frame(mean_row, check.names=FALSE),
        as.data.frame(se_row, check.names=FALSE))
}

update_result <- function(path, repetition, values, tables) {
  table <- numeric_result_rows(read.csv(path, check.names=FALSE))
  row <- which(table$repetition == repetition)
  if (length(row) != 1L) stop("Expected one result row for repetition ", repetition)
  for (column in names(values)) {
    if (!column %in% names(table)) table[[column]] <- NA_real_
    table[[column]][row] <- values[[column]]
  }
  temporary <- paste0(path, ".tmp")
  write.csv(add_summary_rows(table), temporary, row.names=FALSE)
  if (file.exists(path)) file.remove(path)
  if (!file.rename(temporary, path)) stop("Could not replace result table")
  tables$write_density_tables(path)
}

run_hmpt_dataset <- function(
    project_root, dataset, transform, np, repetitions=seq_len(n_repetitions),
    max_K=15L, eta=0.1, NL=32L, particles=1000L,
    max_fit_seconds=Inf) {
  if (!dataset %in% dataset_names) stop("Unknown dataset: ", dataset)
  if (!transform %in% dataset_transforms[[dataset]]) {
    stop("Transform ", transform, " is not configured for ", dataset)
  }
  path <- result_path(project_root, dataset)
  if (!file.exists(path)) stop("Missing result table: generate the data first")
  tables <- reticulate::import_from_path("experiment_utils.tables", path=project_root)
  tables$write_density_tables(path)
  score_column <- paste0("HMPT_", transform, "_log_score")
  time_column <- paste0("HMPT_", transform, "_running_time")
  results <- numeric_result_rows(read.csv(path, check.names=FALSE))
  completed <- if (score_column %in% names(results)) {
    sum(is.finite(suppressWarnings(as.numeric(results[[score_column]]))))
  } else 0L
  cat(sprintf("\n[%s] HMPT_%s: %d/%d repetitions already complete.\n",
              dataset, transform, completed, length(repetitions)))
  flush.console()
  for (repetition in repetitions) {
    row <- which(results$repetition == repetition)
    if (score_column %in% names(results) && length(row)==1L &&
        is.finite(suppressWarnings(as.numeric(results[[score_column]][row])))) {
      cat(sprintf("HMPT_%s %s repetition %02d: completed\n",
                  transform, dataset, repetition))
      next
    }
    x_train <- read_split(project_root, dataset, repetition, "train", np)
    x_test <- read_split(project_root, dataset, repetition, "test", np)
    parameters <- fit_tree_transform(x_train, transform)
    u_train <- apply_tree_transform(x_train, parameters)
    test <- apply_tree_transform(x_test, parameters, jacobian=TRUE)
    cat(sprintf("[%s] HMPT_%s repetition %02d/%02d started at %s.\n",
                dataset, transform, repetition, n_repetitions,
                format(Sys.time(), "%H:%M:%S")))
    flush.console()
    set.seed(2026L + 10000L*repetition + ncol(x_train))
    start <- proc.time()[["elapsed"]]
    fit <- NULL; error_message <- NULL
    setTimeLimit(elapsed=max_fit_seconds, transient=TRUE)
    fit <- tryCatch(
      SMCMP::smc.apt(
        data=u_train, grid.points=test$u, max.resol=max_K,
        n.particle=particles, n.grid.L=NL-1L, eta.R=eta
      ),
      error=function(e) { error_message <<- conditionMessage(e); NULL },
      finally=setTimeLimit(cpu=Inf, elapsed=Inf, transient=FALSE)
    )
    if (is.null(fit)) {
      elapsed <- proc.time()[["elapsed"]] - start
      update_result(path, repetition,
                    setNames(list(NA_real_, elapsed), c(score_column, time_column)), tables)
      cat("  FAILED: ", error_message, "\n", sep="")
      flush.console()
      next
    }
    score <- mean(log(pmax(fit$pred.density, 1e-300)) + test$log_jacobian)
    elapsed <- proc.time()[["elapsed"]] - start
    update_result(path, repetition,
                  setNames(list(score, elapsed), c(score_column, time_column)), tables)
    cat(sprintf("  finished: log_score=%.6f, time=%.1fs\n", score, elapsed))
    flush.console()
    results <- numeric_result_rows(read.csv(path, check.names=FALSE))
  }
  invisible(TRUE)
}
