
import argparse
import json
import os
import urllib.request
import warnings
import zipfile

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; drop this line to view plots interactively
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")  # statsmodels/xgboost emit routine convergence chatter

pd.set_option("display.width", 140)

# Source of the dataset and where it's downloaded/extracted to by default.
DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DEFAULT_DATA_DIR = r"C:\BikeSharingDataset"


# Data Loading

def download_and_extract_dataset(dest_dir: str = DEFAULT_DATA_DIR) -> str:

    os.makedirs(dest_dir, exist_ok=True)

    def _find_hour_csv(root: str):
        for dirpath, _, filenames in os.walk(root):
            if "hour.csv" in filenames:
                return dirpath
        return None

    existing = _find_hour_csv(dest_dir)
    if existing:
        print(f"Dataset already present at {existing}, skipping download.")
        return existing

    zip_path = os.path.join(dest_dir, "bike_sharing_dataset.zip")
    print(f"Downloading dataset from {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    print(f"Downloaded to {zip_path}. Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    os.remove(zip_path)

    found = _find_hour_csv(dest_dir)
    if not found:
        raise FileNotFoundError(
            f"Downloaded and extracted the dataset to {dest_dir}, but could not "
            "locate hour.csv inside it. Check the archive contents manually."
        )
    print(f"Extracted dataset to {found}")
    return found


def load_data(data_dir: str = None):

    if data_dir is None:
        data_dir = download_and_extract_dataset(DEFAULT_DATA_DIR)

    hour_path = os.path.join(data_dir, "hour.csv")
    if not os.path.exists(hour_path):
        raise FileNotFoundError(f"Could not find hour.csv in {data_dir}")
    hour_df = pd.read_csv(hour_path, parse_dates=["dteday"])

    day_path = os.path.join(data_dir, "day.csv")
    day_df = pd.read_csv(day_path, parse_dates=["dteday"]) if os.path.exists(day_path) else None

    print(f"Loaded hour.csv: {hour_df.shape[0]} rows, {hour_df.shape[1]} columns "
          f"({hour_df.dteday.min().date()} to {hour_df.dteday.max().date()})")
    if day_df is not None:
        print(f"Loaded day.csv:  {day_df.shape[0]} rows")
    return hour_df, day_df



# SECTION 1: Descriptive Statistics and EDA

def run_eda(hour_df: pd.DataFrame, output_dir: str):

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print("\n" + "=" * 70 + "\nPART 1 — DESCRIPTIVE STATISTICS & EDA\n" + "=" * 70)

    full_index = pd.MultiIndex.from_product(
        [hour_df.dteday.unique(), range(24)], names=["dteday", "hr"]
    )
    present_index = pd.MultiIndex.from_frame(hour_df[["dteday", "hr"]])
    missing = full_index.difference(present_index)
    print(f"\nMissing hour-slots: {len(missing)} of {len(full_index)}")
    if len(missing) > 0:
        missing_df = pd.DataFrame(missing.tolist(), columns=["dteday", "hr"])
        print("Dates with the most missing hours:")
        print(missing_df["dteday"].value_counts().head(6).to_string())

    #1b. Over-dispersion of the target
    print("\nTarget distribution (mean / variance / var-mean ratio / skew):")
    disp_rows = []
    for col in ["cnt", "casual", "registered"]:
        mean, var = hour_df[col].mean(), hour_df[col].var()
        disp_rows.append({
            "variable": col, "mean": round(mean, 1), "variance": round(var, 1),
            "var_over_mean": round(var / mean, 1), "skew": round(hour_df[col].skew(), 2),
        })
    disp_table = pd.DataFrame(disp_rows)
    print(disp_table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(hour_df.cnt, bins=60, color="#5c7cfa", alpha=0.85)
    ax.axvline(hour_df.cnt.mean(), color="#f76707", ls="--", label=f"mean={hour_df.cnt.mean():.0f}")
    ax.set_xlabel("cnt (rentals per hour)"); ax.set_ylabel("Frequency")
    ax.set_title(f"cnt distribution — skew={hour_df.cnt.skew():.2f}")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(fig_dir, "01_cnt_distribution.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    #1c. Casual vs. registered by hour, split by working day
    hourly = hour_df.groupby(["workingday", "hr"])[["casual", "registered"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, wd, title in zip(axes, [1, 0], ["Working days", "Weekends & holidays"]):
        d = hourly[hourly.workingday == wd]
        ax.plot(d.hr, d.registered, marker="o", ms=3, color="#4c6ef5", label="Registered")
        ax.plot(d.hr, d.casual, marker="o", ms=3, color="#2f9e77", label="Casual")
        ax.set_title(title); ax.set_xlabel("Hour of day")
    axes[0].set_ylabel("Average rentals"); axes[0].legend(frameon=False)
    fig.suptitle("Hourly rental pattern: commuter peaks vs. leisure midday bulge", y=1.03)
    fig.savefig(os.path.join(fig_dir, "02_hourly_pattern.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    casual_share = hour_df.assign(share=hour_df.casual / hour_df.cnt).groupby("workingday")["share"].mean()
    print(f"\nCasual rider share of total demand: working days = {casual_share[1]:.1%}, "
          f"non-working days = {casual_share[0]:.1%}")

    #1d. Weather, season, year, holiday effects
    print("\nMean cnt by weather condition (1=clear ... 4=heavy rain/snow):")
    print(hour_df.groupby("weathersit")["cnt"].agg(["mean", "count"]).round(1).to_string())

    print("\nMean cnt by season (1=winter, 2=spring, 3=summer, 4=fall):")
    print(hour_df.groupby("season")["cnt"].mean().round(1).to_string())

    yr_means = hour_df.groupby("yr")["cnt"].mean()
    print(f"\nYear-over-year growth: {yr_means[0]:.1f} (2011) -> {yr_means[1]:.1f} (2012) "
          f"= +{(yr_means[1] / yr_means[0] - 1):.0%}")

    hol = hour_df.groupby("holiday")[["cnt", "casual", "registered"]].mean().round(1)
    print(f"\nHoliday effect:\n{hol.to_string()}")
    # Net effect is a DECREASE on holidays: lost registered commutes outweigh
    # the gain in casual leisure trips.

    print(f"\nFigures saved to: {fig_dir}")
    return disp_table



# SECTION 2: Regression Analysis

def run_regression(hour_df: pd.DataFrame, output_dir: str):

    print("\n" + "=" * 70 + "\nPART 2 — REGRESSION ANALYSIS\n" + "=" * 70)

    #2a. Verify structural redundancies before choosing variables
    derived_workingday = ((~hour_df.weekday.isin([0, 6])) & (hour_df.holiday == 0)).astype(int)
    match_pct = (derived_workingday == hour_df.workingday).mean() * 100
    print(f"\nworkingday matches weekday+holiday-derived rule for {match_pct:.0f}% of rows "
          "-> workingday is dropped as redundant.")

    #2b. Multicollinearity (VIF) among continuous predictors
    cont_with = sm.add_constant(hour_df[["temp", "atemp", "hum", "windspeed"]])
    vif_with = pd.Series(
        [variance_inflation_factor(cont_with.values, i) for i in range(cont_with.shape[1])],
        index=cont_with.columns,
    )
    cont_without = sm.add_constant(hour_df[["temp", "hum", "windspeed"]])
    vif_without = pd.Series(
        [variance_inflation_factor(cont_without.values, i) for i in range(cont_without.shape[1])],
        index=cont_without.columns,
    )
    print("\nVIF WITH atemp:\n", vif_with.round(2).to_string())
    print("\nVIF WITHOUT atemp:\n", vif_without.round(2).to_string())


    #2c. Fit the model family progression
    formula = "cnt ~ C(mnth) + yr + C(hr) + holiday + C(weekday) + C(weathersit) + temp + hum + windspeed"
    df = hour_df.copy()
    df["log_cnt"] = np.log1p(df.cnt)

    ols_raw = smf.ols(formula, data=df).fit()
    ols_log = smf.ols(formula.replace("cnt ~", "log_cnt ~"), data=df).fit()
    poisson = smf.glm(formula, data=df, family=sm.families.Poisson()).fit()

    # Negative Binomial: estimate the dispersion parameter alpha via MLE,
    # then refit as a GLM with that alpha (statsmodels' standard workflow).
    nb_mle = smf.negativebinomial(formula, data=df).fit(disp=False, maxiter=200)
    alpha_hat = nb_mle.params["alpha"]
    nb = smf.glm(formula, data=df, family=sm.families.NegativeBinomial(alpha=alpha_hat)).fit()

    print("\nModel comparison:")
    print(f"  OLS (raw cnt)     R2={ols_raw.rsquared:.3f}   AIC={ols_raw.aic:,.0f}")
    print(f"  OLS (log1p cnt)   R2={ols_log.rsquared:.3f}   AIC={ols_log.aic:,.0f}  (AIC not comparable across DV transform)")
    print(f"  Poisson GLM       pseudo-R2={1 - poisson.llf / poisson.llnull:.3f}   AIC={poisson.aic:,.0f}")
    print(f"  Negative Binomial pseudo-R2={1 - nb.llf / nb.llnull:.3f}   AIC={nb.aic:,.0f}   alpha={alpha_hat:.3f}")
    print("  -> Negative Binomial wins decisively on AIC (pseudo-R2 is not directly")
    print("     comparable between Poisson and NB, since their null log-likelihoods differ).")

    #2d. Diagnostics justifying the move away from OLS/Poisson
    disp_ratio = poisson.pearson_chi2 / poisson.df_resid
    bp_stat, bp_p, _, _ = sm.stats.diagnostic.het_breuschpagan(ols_raw.resid, ols_raw.model.exog)
    jb_stat, jb_p, _, _ = sm.stats.stattools.jarque_bera(ols_raw.resid)
    print(f"\nPoisson overdispersion (Pearson chi2/df, ~1 if well-specified): {disp_ratio:.1f}")
    print(f"Breusch-Pagan (heteroscedasticity): stat={bp_stat:,.0f}, p={bp_p:.3g}")
    print(f"Jarque-Bera (non-normal residuals): stat={jb_stat:,.0f}, p={jb_p:.3g}")

    #2e. Interpret the Negative Binomial coefficients as IRRs
    irr = np.exp(nb.params)
    irr_ci = np.exp(nb.conf_int())
    print("\nNegative Binomial incidence rate ratios (IRR = e^beta):")
    key_vars = ["yr", "holiday", "temp", "hum", "windspeed",
                "C(weathersit)[T.2]", "C(weathersit)[T.3]", "C(weathersit)[T.4]"]
    irr_rows = []
    for v in key_vars:
        row = {"variable": v, "irr": round(irr[v], 3),
               "ci_low": round(irr_ci.loc[v, 0], 3), "ci_high": round(irr_ci.loc[v, 1], 3),
               "p_value": nb.pvalues[v]}
        irr_rows.append(row)
        print(f"  {v:28s} IRR={row['irr']:.3f}  95% CI=({row['ci_low']:.3f}, {row['ci_high']:.3f})  p={row['p_value']:.3g}")
    irr_table = pd.DataFrame(irr_rows)

    #2f. Save a compact results bundle
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "vif_with_atemp": vif_with.to_dict(), "vif_without_atemp": vif_without.to_dict(),
        "ols_raw_r2": ols_raw.rsquared, "ols_raw_aic": ols_raw.aic,
        "ols_log_r2": ols_log.rsquared, "ols_log_aic": ols_log.aic,
        "poisson_aic": poisson.aic, "poisson_pseudo_r2": 1 - poisson.llf / poisson.llnull,
        "nb_aic": nb.aic, "nb_pseudo_r2": 1 - nb.llf / nb.llnull, "alpha_hat": alpha_hat,
        "overdispersion_ratio": disp_ratio, "breusch_pagan": {"stat": bp_stat, "p": bp_p},
        "jarque_bera": {"stat": jb_stat, "p": jb_p},
    }
    with open(os.path.join(output_dir, "regression_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    irr_table.to_csv(os.path.join(output_dir, "regression_irr_table.csv"), index=False)
    print(f"\nRegression summary saved to: {output_dir}")

    return {"ols_raw": ols_raw, "ols_log": ols_log, "poisson": poisson, "nb": nb, "irr_table": irr_table}


# SECTION 3: Demand Forecasting

def build_features(hour_df: pd.DataFrame) -> pd.DataFrame:

    df = hour_df.sort_values(["dteday", "hr"]).copy()
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")

    # Reindex onto a complete hourly grid so lag/rolling windows are spaced
    # correctly in time (see Part 1: 165 hour-slots are missing, not zero).
    full_idx = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    full = pd.DataFrame({"datetime": full_idx})
    df = full.merge(df, on="datetime", how="left").sort_values("datetime").reset_index(drop=True)

    df["dteday"] = df["datetime"].dt.floor("D")
    df["hr"] = df["datetime"].dt.hour
    df["yr"] = df["datetime"].dt.year - df["datetime"].dt.year.min()
    df["mnth"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.dayofweek.map({6: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6})
    for c in ["season", "holiday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]:
        df[c] = df[c].ffill().bfill()  # short weather/calendar gaps: carry the nearest known value
    for c in ["casual", "registered", "cnt"]:
        df[c] = df[c].fillna(0)  # missing hour-slots treated as zero rentals (documented limitation)
    df["day"] = df["dteday"].dt.day

    df["lag_24"] = df["cnt"].shift(24)
    df["lag_48"] = df["cnt"].shift(48)
    df["lag_168"] = df["cnt"].shift(168)
    df["roll_mean_24"] = df["cnt"].shift(1).rolling(24).mean()
    df["roll_mean_168"] = df["cnt"].shift(1).rolling(168).mean()
    df["roll_std_24"] = df["cnt"].shift(1).rolling(24).std()
    df["reg_roll_mean_168"] = df["registered"].shift(1).rolling(168).mean()
    df["cas_roll_mean_168"] = df["casual"].shift(1).rolling(168).mean()

    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["wd_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["wd_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    return df


FEATURE_COLS = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed", "hr_sin", "hr_cos", "wd_sin", "wd_cos",
    "lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_mean_168", "roll_std_24",
    "reg_roll_mean_168", "cas_roll_mean_168",
]


def _metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None))))
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:24s} RMSE={rmse:7.2f}  MAE={mae:7.2f}  RMSLE={rmsle:.4f}  R2={r2:.4f}")
    return {"name": name, "rmse": rmse, "mae": mae, "rmsle": rmsle, "r2": r2}


def run_forecasting(hour_df: pd.DataFrame, output_dir: str):

    print("\n" + "=" * 70 + "\nPART 3 — DEMAND FORECASTING\n" + "=" * 70)

    df = build_features(hour_df)
    df_model = df.dropna(subset=FEATURE_COLS + ["cnt"]).copy()  # drops ~first 7 days (no lag history yet)
    df_model["is_test"] = df_model["day"] >= 21

    train = df_model[~df_model["is_test"]].copy()
    test = df_model[df_model["is_test"]].copy()
    print(f"\nTrain (days 1-20 of each month):  {len(train)} rows, mean cnt={train.cnt.mean():.1f}")
    print(f"Test  (days 21-end of each month): {len(test)} rows, mean cnt={test.cnt.mean():.1f}")

    X_train, y_train = train[FEATURE_COLS], train["cnt"]
    X_test, y_test = test[FEATURE_COLS], test["cnt"]

    train_sorted = train.sort_values("datetime")
    n_val = int(len(train_sorted) * 0.12)
    tr_fit, tr_val = train_sorted.iloc[:-n_val], train_sorted.iloc[-n_val:]

    def _fit_xgb(log_target: bool):
        ytr = np.log1p(tr_fit["cnt"]) if log_target else tr_fit["cnt"]
        yval = np.log1p(tr_val["cnt"]) if log_target else tr_val["cnt"]
        model = xgb.XGBRegressor(
            n_estimators=2000, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_lambda=1.0, random_state=42, n_jobs=4,
            early_stopping_rounds=50, eval_metric="rmse",
        )
        model.fit(tr_fit[FEATURE_COLS], ytr, eval_set=[(tr_val[FEATURE_COLS], yval)], verbose=False)
        return model.best_iteration

    def _fit_final(n_estimators: int, log_target: bool):
        y = np.log1p(y_train) if log_target else y_train
        model = xgb.XGBRegressor(
            n_estimators=n_estimators, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_lambda=1.0, random_state=42, n_jobs=4,
        )
        model.fit(X_train, y)
        return model

    best_it_raw = _fit_xgb(log_target=False)
    best_it_log = _fit_xgb(log_target=True)
    final_raw = _fit_final(best_it_raw + 1, log_target=False)
    final_log = _fit_final(best_it_log + 1, log_target=True)

    pred_naive = X_test["lag_168"].values  # seasonal-naive baseline: "same as last week"
    pred_raw = np.clip(final_raw.predict(X_test), 0, None)
    pred_log = np.clip(np.expm1(final_log.predict(X_test)), 0, None)

    print("\nTest-set performance:")
    results = [
        _metrics(y_test, pred_naive, "Naive (lag 168h)"),
        _metrics(y_test, pred_raw, "XGBoost, raw target"),
        _metrics(y_test, pred_log, "XGBoost, log1p target"),
    ]

    # Feature importance (log1p model)
    importance = pd.Series(final_log.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop 8 features by importance (XGBoost, log1p target):")
    print(importance.head(8).round(4).to_string())

    # Per-month breakdown, normalized by that month's mean demand so winter's
    # naturally lower volumes are compared fairly against summer's.
    test_eval = test.copy()
    test_eval["pred"] = pred_log
    test_eval["mnth_yr"] = test_eval["dteday"].dt.to_period("M").astype(str)
    monthly = (
        test_eval.groupby("mnth_yr")
        .apply(lambda d: pd.Series({
            "n": len(d),
            "mean_actual": d["cnt"].mean(),
            "mae": mean_absolute_error(d["cnt"], d["pred"]),
            "rmse": np.sqrt(mean_squared_error(d["cnt"], d["pred"])),
        }))
        .reset_index()
    )
    monthly["norm_mae_pct"] = (monthly["mae"] / monthly["mean_actual"] * 100).round(1)
    print("\nWorst 3 evaluation windows (by normalized MAE):")
    print(monthly.nlargest(3, "norm_mae_pct")[["mnth_yr", "mean_actual", "mae", "norm_mae_pct"]].to_string(index=False))
    print("\nBest 3 evaluation windows (by normalized MAE):")
    print(monthly.nsmallest(3, "norm_mae_pct")[["mnth_yr", "mean_actual", "mae", "norm_mae_pct"]].to_string(index=False))

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "forecasting_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    monthly.to_csv(os.path.join(output_dir, "forecasting_monthly_breakdown.csv"), index=False)
    importance.to_csv(os.path.join(output_dir, "forecasting_feature_importance.csv"))
    print(f"\nForecasting results saved to: {output_dir}")

    return {"results": results, "monthly": monthly, "importance": importance, "test_eval": test_eval}


# Main
def main():
    parser = argparse.ArgumentParser(description="Capital Bikeshare demand analysis pipeline")
    parser.add_argument(
        "--data-dir", default=None,
        help=f"Directory containing hour.csv (and optionally day.csv). "
             f"If omitted, the dataset is auto-downloaded to {DEFAULT_DATA_DIR}.",
    )
    parser.add_argument("--output-dir", default="./outputs", help="Directory to write figures/tables/metrics to")
    args = parser.parse_args()

    hour_df, day_df = load_data(args.data_dir)
    run_eda(hour_df, args.output_dir)
    run_regression(hour_df, args.output_dir)
    run_forecasting(hour_df, args.output_dir)

    print("\n" + "=" * 70 + f"\nDone. All outputs written to: {os.path.abspath(args.output_dir)}\n" + "=" * 70)


if __name__ == "__main__":
    main()
