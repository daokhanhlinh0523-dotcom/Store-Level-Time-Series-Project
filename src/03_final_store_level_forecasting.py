# %%
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from IPython.display import display
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 140)

CWD = Path.cwd()
if (CWD / 'processed_data').exists():
    NOTEBOOK_DIR = CWD
    REPO_DIR = CWD.parent
elif (CWD / 'notebooks' / 'processed_data').exists():
    REPO_DIR = CWD
    NOTEBOOK_DIR = CWD / 'notebooks'
else:
    raise FileNotFoundError('Cannot locate notebooks/processed_data from the current working directory.')

DATA_DIR = NOTEBOOK_DIR / 'processed_data'
OUTPUT_DIR = REPO_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid')

print('REPO_DIR   :', REPO_DIR)
print('NOTEBOOK_DIR:', NOTEBOOK_DIR)
print('processed_data exists:', DATA_DIR.exists())
print('output exists        :', OUTPUT_DIR.exists())

# %%
visits = pd.read_csv(DATA_DIR / 'air_visits.csv')
air_reserves = pd.read_csv(DATA_DIR / 'air_reserve.csv')
hpg_reserves = pd.read_csv(DATA_DIR / 'hpg_reserve.csv')
store_ids_map = pd.read_csv(DATA_DIR / 'store_ids.csv')
holidays = pd.read_csv(DATA_DIR / 'holidays.csv')
stores = pd.read_csv(DATA_DIR / 'air_store.csv')

visits['visit_date'] = pd.to_datetime(visits['visit_date'])
visits['visitors'] = pd.to_numeric(visits['visitors'])

air_reserves['visit_datetime'] = pd.to_datetime(air_reserves['visit_datetime'])
air_reserves['reserve_datetime'] = pd.to_datetime(air_reserves['reserve_datetime'])
air_reserves['air_store_id'] = air_reserves['air_store_id'].astype(str)

hpg_reserves['visit_datetime'] = pd.to_datetime(hpg_reserves['visit_datetime'])
hpg_reserves['reserve_datetime'] = pd.to_datetime(hpg_reserves['reserve_datetime'])
hpg_reserves = hpg_reserves.merge(store_ids_map, on='hpg_store_id', how='inner')
hpg_reserves = hpg_reserves[['air_store_id', 'visit_datetime', 'reserve_datetime', 'reserve_visitors']]

reserves = pd.concat([air_reserves, hpg_reserves], ignore_index=True)
reserves = reserves.loc[reserves['reserve_datetime'] <= reserves['visit_datetime']].copy()
reserves['visit_date'] = reserves['visit_datetime'].dt.normalize()

holiday_date_col = 'visit_date' if 'visit_date' in holidays.columns else 'calendar_date'
holidays['visit_date'] = pd.to_datetime(holidays[holiday_date_col])
if 'holiday_flg' in holidays.columns and 'is_holiday' not in holidays.columns:
    holidays['is_holiday'] = holidays['holiday_flg']

holidays['is_holiday'] = pd.to_numeric(holidays['is_holiday']).astype(float)
holidays['is_weekend'] = holidays['visit_date'].dt.dayofweek.isin([5, 6]).astype(float)
holidays['is_golden_week'] = (
    (((holidays['visit_date'].dt.month == 4) & (holidays['visit_date'].dt.day >= 29)) |
     ((holidays['visit_date'].dt.month == 5) & (holidays['visit_date'].dt.day <= 5))).astype(float)
)

print('visits        :', visits.shape)
print('air_reserves  :', air_reserves.shape)
print('hpg_reserves  :', hpg_reserves.shape)
print('combined_resv :', reserves.shape)
print('holidays      :', holidays.shape)
print('stores        :', stores.shape)
display(visits.head())

# %%
def seasonal_naive_forecast(train_vals: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    hist = list(train_vals)
    preds = []
    for _ in range(horizon):
        pred = hist[-season]
        preds.append(pred)
        hist.append(pred)
    return np.asarray(preds, dtype=float)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def sanitize_forecast_outputs(
    preds_log: np.ndarray,
    model_label: str,
    max_abs_log: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    preds_log = np.asarray(preds_log, dtype=float)
    if preds_log.ndim != 1:
        raise ValueError(f'{model_label}: forecast must be 1D, got shape {preds_log.shape}')
    if not np.isfinite(preds_log).all():
        raise ValueError(f'{model_label}: forecast contains non-finite log predictions')
    if np.any(np.abs(preds_log) > max_abs_log):
        raise ValueError(f'{model_label}: forecast log predictions exceeded +/-{max_abs_log}')

    preds = np.expm1(preds_log)
    if not np.isfinite(preds).all():
        raise ValueError(f'{model_label}: forecast contains non-finite visitor predictions after expm1')

    preds = np.clip(preds, 0.0, None)
    preds_log = np.log1p(preds)
    return preds_log, preds


def nonnegative_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100.0)


def build_reserve_features(reserves: pd.DataFrame, cutoff_datetime: pd.Timestamp | None = None) -> pd.DataFrame:
    df = reserves.copy()
    if cutoff_datetime is not None:
        df = df.loc[df['reserve_datetime'] <= cutoff_datetime].copy()
    reserve_agg = (
        df.groupby(['air_store_id', 'visit_date'], as_index=False)
        .agg(total_reserve_visitors=('reserve_visitors', 'sum'))
    )
    reserve_agg['log1p_reserve'] = np.log1p(reserve_agg['total_reserve_visitors'])
    return reserve_agg


def build_store_daily_frame(
    store_id: str,
    visits: pd.DataFrame,
    reserve_agg_full: pd.DataFrame,
    reserve_agg_capped: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    store_visits = visits.loc[visits['air_store_id'] == store_id, ['visit_date', 'visitors']].copy()
    store_visits = store_visits.sort_values('visit_date')
    start_date = store_visits['visit_date'].min()
    end_date = store_visits['visit_date'].max()

    grid = pd.DataFrame({'visit_date': pd.date_range(start_date, end_date, freq='D')})
    df = grid.merge(store_visits, on='visit_date', how='left')

    df['visitors'] = df['visitors'].fillna(0.0).astype(float)
    df['log1p_visitors'] = np.log1p(df['visitors'])

    df = df.merge(
        holidays[['visit_date', 'is_holiday', 'is_weekend', 'is_golden_week']],
        on='visit_date',
        how='left',
    )
    df[['is_holiday', 'is_weekend', 'is_golden_week']] = df[['is_holiday', 'is_weekend', 'is_golden_week']].fillna(0.0).astype(float)

    full_store_reserve = reserve_agg_full.loc[reserve_agg_full['air_store_id'] == store_id, ['visit_date', 'log1p_reserve']]
    capped_store_reserve = reserve_agg_capped.loc[reserve_agg_capped['air_store_id'] == store_id, ['visit_date', 'log1p_reserve']]

    df = df.merge(full_store_reserve.rename(columns={'log1p_reserve': 'log1p_reserve_train'}), on='visit_date', how='left')
    df = df.merge(capped_store_reserve.rename(columns={'log1p_reserve': 'log1p_reserve_test'}), on='visit_date', how='left')
    df['log1p_reserve_train'] = df['log1p_reserve_train'].fillna(0.0).astype(float)
    df['log1p_reserve_test'] = df['log1p_reserve_test'].fillna(0.0).astype(float)
    df['air_store_id'] = store_id
    return df


EXOG_CONFIGS = {
    'sarima_no_exog': [],
    'arimax_holiday_only': ['is_holiday'],
}

MODEL_LABELS = {
    'seasonal_naive': 'Seasonal Naive (lag 7)',
    'sarima_no_exog': 'SARIMA no exog',
    'arimax_holiday_only': 'SARIMAX holiday only',
}

FEATURE_SET_LABELS = {
    'seasonal_naive': 'lag 7 only',
    'sarima_no_exog': 'no exog',
    'arimax_holiday_only': 'holiday only',
}

PRIMARY_MODEL_KEYS = ['seasonal_naive', 'sarima_no_exog', 'arimax_holiday_only']


def build_exog(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    feature_cols = feature_cols or []
    if not feature_cols:
        return None, None

    train_col_map = {
        'is_holiday': 'is_holiday',
        'is_golden_week': 'is_golden_week',
        'is_weekend': 'is_weekend',
        'log1p_reserve': 'log1p_reserve_train',
    }
    test_col_map = {
        'is_holiday': 'is_holiday',
        'is_golden_week': 'is_golden_week',
        'is_weekend': 'is_weekend',
        'log1p_reserve': 'log1p_reserve_test',
    }

    x_train = train[[train_col_map[col] for col in feature_cols]].copy()
    x_test = test[[test_col_map[col] for col in feature_cols]].copy()
    x_train.columns = feature_cols
    x_test.columns = feature_cols
    return x_train.astype(float), x_test.astype(float)


def fit_sarimax_forecast(
    y_train: pd.Series,
    horizon: int,
    x_train: pd.DataFrame | None = None,
    x_test: pd.DataFrame | None = None,
    model_label: str = 'SARIMAX',
) -> tuple[np.ndarray, float, str]:
    fit = SARIMAX(
        y_train,
        exog=x_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=50)
    if x_test is None:
        preds_log = np.asarray(fit.forecast(steps=horizon), dtype=float)
    else:
        preds_log = np.asarray(fit.forecast(steps=horizon, exog=x_test), dtype=float)
    preds_log, preds = sanitize_forecast_outputs(preds_log, model_label=model_label)
    return preds_log, preds, float(fit.aic), model_label


def evaluate_store_split(
    store_id: str,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    reserve_agg_full: pd.DataFrame,
    reserve_agg_capped: pd.DataFrame,
    min_train_days: int = 90,
) -> tuple[dict | None, pd.DataFrame | None, dict | None]:
    df = build_store_daily_frame(store_id, visits, reserve_agg_full, reserve_agg_capped, holidays)
    train = df.loc[df['visit_date'] <= train_end].copy()
    test = df.loc[(df['visit_date'] >= test_start) & (df['visit_date'] <= test_end)].copy()

    if len(train) < min_train_days or len(test) == 0:
        return None, None, {
            'air_store_id': store_id,
            'train_days': int(len(train)),
            'test_days': int(len(test)),
            'reason': 'insufficient_train_or_no_test_rows',
        }

    y_train = train['log1p_visitors'].astype(float)
    y_test = test['visitors'].astype(float).to_numpy()
    y_test_log = test['log1p_visitors'].astype(float).to_numpy()

    baseline_log = seasonal_naive_forecast(y_train.to_numpy(), len(test), season=7)
    baseline_log, baseline_pred = sanitize_forecast_outputs(baseline_log, model_label='Seasonal Naive')

    model_logs = {'seasonal_naive': baseline_log}
    model_preds = {'seasonal_naive': baseline_pred}
    model_aic = {'seasonal_naive': np.nan}
    model_status = {'seasonal_naive': 'Seasonal Naive'}

    for model_key, feature_cols in EXOG_CONFIGS.items():
        x_train, x_test = build_exog(train, test, feature_cols)
        try:
            preds_log, preds, fit_aic, fit_status = fit_sarimax_forecast(
                y_train,
                horizon=len(test),
                x_train=x_train,
                x_test=x_test,
                model_label=MODEL_LABELS[model_key],
            )
        except Exception as exc:
            preds_log = baseline_log.copy()
            preds = baseline_pred.copy()
            fit_aic = np.nan
            fit_status = f"{MODEL_LABELS[model_key]}_fallback_to_baseline: {exc}"

        model_logs[model_key] = preds_log
        model_preds[model_key] = preds
        model_aic[model_key] = fit_aic
        model_status[model_key] = fit_status

    metrics_row = {
        'air_store_id': store_id,
        'train_days': int(len(train)),
        'test_days': int(len(test)),
        'seasonal_naive_rmsle': rmsle(y_test, model_preds['seasonal_naive']),
        'seasonal_naive_rmse_log': float(np.sqrt(np.mean((y_test_log - model_logs['seasonal_naive']) ** 2))),
        'seasonal_naive_mape': nonnegative_mape(y_test, model_preds['seasonal_naive']),
    }

    for model_key in EXOG_CONFIGS:
        metrics_row[f'{model_key}_rmsle'] = rmsle(y_test, model_preds[model_key])
        metrics_row[f'{model_key}_rmse_log'] = float(np.sqrt(np.mean((y_test_log - model_logs[model_key]) ** 2)))
        metrics_row[f'{model_key}_mape'] = nonnegative_mape(y_test, model_preds[model_key])
        metrics_row[f'{model_key}_aic'] = model_aic[model_key]
        metrics_row[f'{model_key}_status'] = model_status[model_key]

    pred_chunk = pd.DataFrame({
        'air_store_id': store_id,
        'visit_date': test['visit_date'].dt.strftime('%Y-%m-%d'),
        'actual_visitors': y_test,
        'actual_log1p': y_test_log,
    })

    for model_key, preds in model_preds.items():
        pred_chunk[f'{model_key}_pred'] = preds
        pred_chunk[f'{model_key}_log1p'] = model_logs[model_key]

    return metrics_row, pred_chunk, None


def evaluate_split(
    split_name: str,
    train_end: str,
    test_start: str,
    test_end: str,
    reserve_agg_full: pd.DataFrame,
    run_store_ids: list[str],
    min_train_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)
    capped_cutoff = train_end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    reserve_agg_capped = build_reserve_features(reserves, cutoff_datetime=capped_cutoff)

    rows = []
    chunks = []
    skipped = []

    for store_id in run_store_ids:
        row, chunk, skip = evaluate_store_split(
            store_id,
            train_end=train_end_ts,
            test_start=test_start_ts,
            test_end=test_end_ts,
            reserve_agg_full=reserve_agg_full,
            reserve_agg_capped=reserve_agg_capped,
            min_train_days=min_train_days,
        )
        if skip is not None:
            skipped.append(skip)
            continue
        rows.append(row)
        chunks.append(chunk)

    metrics = pd.DataFrame(rows).merge(store_meta, on='air_store_id', how='left')
    preds = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped)

    def pooled_model_rmsle(model_key: str) -> float:
        return rmsle(preds['actual_visitors'], preds[f'{model_key}_pred']) if len(preds) else np.nan

    summary = {
        'split': split_name,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'n_stores_evaluated': int(len(metrics)),
        'n_stores_skipped': int(len(skipped_df)),
        'pooled_rows': int(len(preds)),
        'seasonal_naive_rmsle': pooled_model_rmsle('seasonal_naive'),
        'sarima_no_exog_rmsle': pooled_model_rmsle('sarima_no_exog'),
        'arimax_holiday_only_rmsle': pooled_model_rmsle('arimax_holiday_only'),
        'sarima_better_than_naive_share': float((metrics['sarima_no_exog_rmsle'] < metrics['seasonal_naive_rmsle']).mean()) if len(metrics) else np.nan,
        'holiday_arimax_better_than_naive_share': float((metrics['arimax_holiday_only_rmsle'] < metrics['seasonal_naive_rmsle']).mean()) if len(metrics) else np.nan,
        'holiday_arimax_better_than_sarima_share': float((metrics['arimax_holiday_only_rmsle'] < metrics['sarima_no_exog_rmsle']).mean()) if len(metrics) else np.nan,
    }
    summary['sarima_improvement_vs_naive_pct'] = 100.0 * (summary['seasonal_naive_rmsle'] - summary['sarima_no_exog_rmsle']) / summary['seasonal_naive_rmsle']
    summary['holiday_arimax_improvement_vs_naive_pct'] = 100.0 * (summary['seasonal_naive_rmsle'] - summary['arimax_holiday_only_rmsle']) / summary['seasonal_naive_rmsle']
    summary['holiday_arimax_improvement_vs_sarima_pct'] = 100.0 * (summary['sarima_no_exog_rmsle'] - summary['arimax_holiday_only_rmsle']) / summary['sarima_no_exog_rmsle']
    return metrics, preds, skipped_df, summary


print('Helpers loaded.')

# %%
final_split_date = pd.Timestamp('2017-03-14')
final_holdout_start = pd.Timestamp('2017-03-15')
final_holdout_end = pd.Timestamp('2017-04-22')
reserve_agg_full = build_reserve_features(reserves)
reserve_agg_final = build_reserve_features(
    reserves,
    cutoff_datetime=final_split_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1),
)
store_meta = stores[['air_store_id', 'air_genre_name', 'air_area_name']].copy()
store_ids = sorted(visits['air_store_id'].unique())

observed_span = visits.groupby('air_store_id')['visit_date'].agg(['min', 'max', 'size']).reset_index()
observed_span['prehistory_days_removed'] = (observed_span['min'] - visits['visit_date'].min()).dt.days.clip(lower=0)
observed_span['train_days_available'] = ((final_split_date - observed_span['min']).dt.days + 1).clip(lower=0)
observed_span['has_holdout_window'] = observed_span['max'] >= final_holdout_start

min_train_days = 90

RUN_MODE = 'main_sample'
SANITY_CHECK_STORE_LIMIT = 250
OUTPUT_TAG = 'main250cv'

eligible_store_ids = (
    observed_span.loc[
        (observed_span['train_days_available'] >= min_train_days) & observed_span['has_holdout_window'],
        ['air_store_id', 'size', 'train_days_available']
    ]
    .sort_values(['size', 'train_days_available', 'air_store_id'], ascending=[False, False, True])
)

run_store_ids = eligible_store_ids.head(SANITY_CHECK_STORE_LIMIT)['air_store_id'].tolist()

cv_folds = [
    {'split': 'CV1', 'train_end': '2017-01-14', 'test_start': '2017-01-15', 'test_end': '2017-01-28'},
    {'split': 'CV2', 'train_end': '2017-01-31', 'test_start': '2017-02-01', 'test_end': '2017-02-14'},
    {'split': 'CV3', 'train_end': '2017-02-14', 'test_start': '2017-02-15', 'test_end': '2017-02-28'},
]

print('run mode                   :', RUN_MODE)
print('output tag                 :', OUTPUT_TAG)
print('store limit                :', SANITY_CHECK_STORE_LIMIT)
print('n_stores total             :', len(store_ids))
print('n_stores eligible          :', len(eligible_store_ids))
print('n_stores selected          :', len(run_store_ids))
print('global visit range         :', visits['visit_date'].min().date(), '->', visits['visit_date'].max().date())
print('final holdout              :', final_holdout_start.date(), '->', final_holdout_end.date())
print('reserve_agg_full           :', reserve_agg_full.shape)
print('reserve_agg_final          :', reserve_agg_final.shape)
print('stores with removed padding:', int((observed_span['prehistory_days_removed'] > 0).sum()))
display(observed_span.head())
display(eligible_store_ids.head(10))
display(pd.DataFrame(cv_folds))

# %%
sample_store_id = run_store_ids[0]
sample_df = build_store_daily_frame(sample_store_id, visits, reserve_agg_full, reserve_agg_final, holidays)
sample_train = sample_df.loc[sample_df['visit_date'] <= final_split_date].copy()
sample_test = sample_df.loc[sample_df['visit_date'] > final_split_date].copy()

print('sample_store_id :', sample_store_id)
print('sample_df shape :', sample_df.shape)
print('train/test days :', len(sample_train), len(sample_test))
display(sample_df.head(10))

# %%
y_train = sample_train['log1p_visitors'].astype(float)
y_test = sample_test['visitors'].astype(float).to_numpy()

baseline_log = seasonal_naive_forecast(y_train.to_numpy(), len(sample_test), season=7)
baseline_log, baseline_pred = sanitize_forecast_outputs(baseline_log, model_label='Seasonal Naive sample')

sample_model_outputs = {
    'seasonal_naive': baseline_pred,
}

sample_sarima_log, sample_sarima_pred, sample_sarima_aic, _ = fit_sarimax_forecast(
    y_train,
    horizon=len(sample_test),
    model_label='SARIMA no exog',
)
sample_model_outputs['sarima_no_exog'] = sample_sarima_pred

sample_x_train, sample_x_test = build_exog(sample_train, sample_test, EXOG_CONFIGS['arimax_holiday_only'])
sample_arimax_log, sample_arimax_pred, sample_arimax_aic, _ = fit_sarimax_forecast(
    y_train,
    horizon=len(sample_test),
    x_train=sample_x_train,
    x_test=sample_x_test,
    model_label='SARIMAX holiday only',
)
sample_model_outputs['arimax_holiday_only'] = sample_arimax_pred

sample_metrics = pd.DataFrame([
    {
        'air_store_id': sample_store_id,
        'train_days': len(sample_train),
        'test_days': len(sample_test),
        'seasonal_naive_rmsle': rmsle(y_test, sample_model_outputs['seasonal_naive']),
        'sarima_no_exog_rmsle': rmsle(y_test, sample_model_outputs['sarima_no_exog']),
        'arimax_holiday_only_rmsle': rmsle(y_test, sample_model_outputs['arimax_holiday_only']),
        'sarima_no_exog_aic': sample_sarima_aic,
        'arimax_holiday_only_aic': sample_arimax_aic,
    }
])

display(sample_metrics)
display(pd.DataFrame({
    'visit_date': sample_test['visit_date'].dt.strftime('%Y-%m-%d'),
    'actual_visitors': y_test,
    'seasonal_naive_pred': sample_model_outputs['seasonal_naive'],
    'sarima_no_exog_pred': sample_model_outputs['sarima_no_exog'],
    'arimax_holiday_only_pred': sample_model_outputs['arimax_holiday_only'],
}).head(10))

# %%
cv_summaries = []
cv_metrics_frames = []
cv_pred_frames = []
cv_skipped_frames = []

for fold in cv_folds:
    print(f"Running {fold['split']} ...")
    fold_metrics, fold_preds, fold_skipped, fold_summary = evaluate_split(
        split_name=fold['split'],
        train_end=fold['train_end'],
        test_start=fold['test_start'],
        test_end=fold['test_end'],
        reserve_agg_full=reserve_agg_full,
        run_store_ids=run_store_ids,
        min_train_days=min_train_days,
    )
    cv_summaries.append(fold_summary)
    if len(fold_metrics):
        cv_metrics_frames.append(fold_metrics.assign(split=fold['split']))
    if len(fold_preds):
        cv_pred_frames.append(fold_preds.assign(split=fold['split']))
    if len(fold_skipped):
        cv_skipped_frames.append(fold_skipped.assign(split=fold['split']))

cv_summary_df = pd.DataFrame(cv_summaries)
cv_store_metrics = pd.concat(cv_metrics_frames, ignore_index=True) if cv_metrics_frames else pd.DataFrame()
cv_predictions = pd.concat(cv_pred_frames, ignore_index=True) if cv_pred_frames else pd.DataFrame()
cv_skipped = pd.concat(cv_skipped_frames, ignore_index=True) if cv_skipped_frames else pd.DataFrame()

display(cv_summary_df)

# %%
rows = []
pooled_chunks = []
failures = []
skipped_stores = []

for idx, store_id in enumerate(run_store_ids, start=1):
    row, chunk, skip = evaluate_store_split(
        store_id,
        train_end=final_split_date,
        test_start=final_holdout_start,
        test_end=final_holdout_end,
        reserve_agg_full=reserve_agg_full,
        reserve_agg_capped=reserve_agg_final,
        min_train_days=min_train_days,
    )
    if skip is not None:
        skipped_stores.append(skip)
        continue

    rows.append(row)
    pooled_chunks.append(chunk)

    for model_key in EXOG_CONFIGS:
        status_value = row.get(f'{model_key}_status')
        if isinstance(status_value, str) and 'fallback_to_baseline' in status_value:
            failures.append({'air_store_id': store_id, 'model': model_key, 'error': status_value})

    if idx % 25 == 0 or idx == len(run_store_ids):
        print(f'done {idx} / {len(run_store_ids)} stores')

store_metrics = pd.DataFrame(rows).merge(store_meta, on='air_store_id', how='left')
pooled_df = pd.concat(pooled_chunks, ignore_index=True) if pooled_chunks else pd.DataFrame()
skipped_df = pd.DataFrame(skipped_stores)
failures_df = pd.DataFrame(failures)

print('store_metrics shape :', store_metrics.shape)
print('pooled_df shape     :', pooled_df.shape)
print('skipped stores      :', len(skipped_stores))
print('model fallbacks     :', len(failures))
display(store_metrics.head())

# %%
def pooled_metric_table(pooled_df: pd.DataFrame) -> pd.DataFrame:
    metric_rows = []
    model_order = ['seasonal_naive'] + list(EXOG_CONFIGS.keys())
    for model_key in model_order:
        metric_rows.append({
            'model_key': model_key,
            'model': MODEL_LABELS[model_key],
            'feature_set': FEATURE_SET_LABELS[model_key],
            'scope': 'pooled observed store-date holdout',
            'RMSLE': rmsle(pooled_df['actual_visitors'], pooled_df[f'{model_key}_pred']),
            'RMSE_log': float(np.sqrt(np.mean((pooled_df['actual_log1p'] - pooled_df[f'{model_key}_log1p']) ** 2))),
            'MAPE': nonnegative_mape(pooled_df['actual_visitors'], pooled_df[f'{model_key}_pred']),
        })
    return pd.DataFrame(metric_rows)


final_metrics = pooled_metric_table(pooled_df)
metrics_lookup = final_metrics.set_index('model_key')

summary = {
    'run_mode': RUN_MODE,
    'output_tag': OUTPUT_TAG,
    'n_stores_total': int(len(store_ids)),
    'n_stores_eligible': int(len(eligible_store_ids)),
    'n_stores_targeted': int(len(run_store_ids)),
    'n_stores_evaluated': int(len(store_metrics)),
    'n_stores_skipped': int(len(skipped_stores)),
    'n_model_fallbacks': int(len(failures)),
    'holdout_start': str(final_holdout_start.date()),
    'holdout_end': str(final_holdout_end.date()),
    'pooled_rows': int(len(pooled_df)),
    'seasonal_naive_rmsle_pooled': float(metrics_lookup.loc['seasonal_naive', 'RMSLE']),
    'sarima_no_exog_rmsle_pooled': float(metrics_lookup.loc['sarima_no_exog', 'RMSLE']),
    'arimax_holiday_only_rmsle_pooled': float(metrics_lookup.loc['arimax_holiday_only', 'RMSLE']),
    'sarima_better_than_naive_share': float((store_metrics['sarima_no_exog_rmsle'] < store_metrics['seasonal_naive_rmsle']).mean()),
    'holiday_arimax_better_than_naive_share': float((store_metrics['arimax_holiday_only_rmsle'] < store_metrics['seasonal_naive_rmsle']).mean()),
    'holiday_arimax_better_than_sarima_share': float((store_metrics['arimax_holiday_only_rmsle'] < store_metrics['sarima_no_exog_rmsle']).mean()),
}
summary['sarima_improvement_vs_naive_pct'] = 100.0 * (
    summary['seasonal_naive_rmsle_pooled'] - summary['sarima_no_exog_rmsle_pooled']
) / summary['seasonal_naive_rmsle_pooled']
summary['holiday_arimax_improvement_vs_naive_pct'] = 100.0 * (
    summary['seasonal_naive_rmsle_pooled'] - summary['arimax_holiday_only_rmsle_pooled']
) / summary['seasonal_naive_rmsle_pooled']
summary['holiday_arimax_improvement_vs_sarima_pct'] = 100.0 * (
    summary['sarima_no_exog_rmsle_pooled'] - summary['arimax_holiday_only_rmsle_pooled']
) / summary['sarima_no_exog_rmsle_pooled']

summary_df = pd.DataFrame([summary]).T.rename(columns={0: 'value'})

store_metrics['delta_sarima_vs_naive'] = store_metrics['seasonal_naive_rmsle'] - store_metrics['sarima_no_exog_rmsle']
store_metrics['delta_holiday_arimax_vs_naive'] = store_metrics['seasonal_naive_rmsle'] - store_metrics['arimax_holiday_only_rmsle']
store_metrics['delta_holiday_arimax_vs_sarima'] = store_metrics['sarima_no_exog_rmsle'] - store_metrics['arimax_holiday_only_rmsle']

display(summary_df)
display(final_metrics[['model', 'feature_set', 'RMSLE', 'RMSE_log', 'MAPE']])
display(cv_summary_df)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

sns.histplot(store_metrics['delta_holiday_arimax_vs_naive'], bins=30, ax=axes[0], color='#1f77b4')
axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
axes[0].set_title('Store-level RMSLE gain: holiday SARIMAX vs baseline')
axes[0].set_xlabel('Baseline RMSLE - holiday SARIMAX RMSLE')
axes[0].set_ylabel('Number of stores')

sns.barplot(data=final_metrics, x='model', y='RMSLE', ax=axes[1], palette=['#9aa0a6', '#4e79a7', '#1f77b4'])
axes[1].set_title('Pooled holdout comparison')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('RMSLE')
axes[1].tick_params(axis='x', rotation=12)

cv_plot_df = cv_summary_df.melt(
    id_vars=['split'],
    value_vars=['seasonal_naive_rmsle', 'sarima_no_exog_rmsle', 'arimax_holiday_only_rmsle'],
    var_name='model_key',
    value_name='rmsle',
)
cv_plot_df['model'] = cv_plot_df['model_key'].map({
    'seasonal_naive_rmsle': 'Seasonal Naive',
    'sarima_no_exog_rmsle': 'SARIMA no exog',
    'arimax_holiday_only_rmsle': 'SARIMAX holiday only',
})
sns.barplot(data=cv_plot_df, x='split', y='rmsle', hue='model', ax=axes[2], palette=['#9aa0a6', '#4e79a7', '#1f77b4'])
axes[2].set_title('3-fold rolling-origin RMSLE')
axes[2].set_xlabel('Fold')
axes[2].set_ylabel('Pooled RMSLE')
axes[2].legend(title='Model')

plt.tight_layout()
plt.show()

representative = {
    'Best gain': store_metrics.sort_values('delta_holiday_arimax_vs_naive', ascending=False).iloc[0]['air_store_id'],
    'Median case': store_metrics.iloc[(store_metrics['delta_holiday_arimax_vs_naive'] - store_metrics['delta_holiday_arimax_vs_naive'].median()).abs().argsort().iloc[0]]['air_store_id'],
    'Holiday SARIMAX worse': store_metrics.sort_values('delta_holiday_arimax_vs_naive', ascending=True).iloc[0]['air_store_id'],
}

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
for ax, (label, store_id) in zip(axes, representative.items()):
    store_plot = pooled_df.loc[pooled_df['air_store_id'] == store_id].copy()
    store_plot['visit_date'] = pd.to_datetime(store_plot['visit_date'])
    ax.plot(store_plot['visit_date'], store_plot['actual_visitors'], label='Actual', color='black', linewidth=2)
    ax.plot(store_plot['visit_date'], store_plot['seasonal_naive_pred'], label='Seasonal Naive', color='#9aa0a6')
    ax.plot(store_plot['visit_date'], store_plot['sarima_no_exog_pred'], label='SARIMA no exog', color='#4e79a7')
    ax.plot(store_plot['visit_date'], store_plot['arimax_holiday_only_pred'], label='SARIMAX holiday only', color='#1f77b4')
    delta = float(store_metrics.loc[store_metrics['air_store_id'] == store_id, 'delta_holiday_arimax_vs_naive'].iloc[0])
    ax.set_title(f'{label}: {store_id} | delta RMSLE = {delta:.3f}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Visitors')
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

cv_summary_df.to_csv(OUTPUT_DIR / f'store_level_cv_summary_{OUTPUT_TAG}.csv', index=False)
store_metrics.to_csv(OUTPUT_DIR / f'store_level_metrics_{OUTPUT_TAG}.csv', index=False)
pooled_df.to_csv(OUTPUT_DIR / f'store_level_holdout_predictions_{OUTPUT_TAG}.csv', index=False)
final_metrics.to_csv(OUTPUT_DIR / f'store_level_ablation_metrics_{OUTPUT_TAG}.csv', index=False)

with open(OUTPUT_DIR / f'store_level_summary_{OUTPUT_TAG}.json', 'w', encoding='utf-8') as f:
    json.dump(
        {
            **summary,
            'comparison_results': final_metrics[['model', 'feature_set', 'RMSLE', 'RMSE_log', 'MAPE']].to_dict(orient='records'),
            'cv_folds': cv_summary_df.to_dict(orient='records'),
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f'Saved updated outputs to ../output/store_level_*_{OUTPUT_TAG}.*')

# %%
from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd
RUN_ROBUSTNESS_400 = False
ROBUSTNESS_STORE_LIMIT = 400
ROBUSTNESS_OUTPUT_TAG = "robust400holdout"

RUN_REPRESENTATIVE_RESIDUAL_DIAGNOSTICS = False
RESIDUAL_DIAGNOSTIC_OUTPUT_TAG = "representative_residual_diagnostics"
RESIDUAL_DIAGNOSTIC_LAGS = [7, 14, 21]

def run_panel_evaluation(
    selected_store_ids: list[str],
    output_tag: str,
    run_cv: bool = False,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv_summary_local = pd.DataFrame()
    if run_cv:
        cv_summaries_local = []
        for fold in cv_folds:
            _, _, _, fold_summary = evaluate_split(
                split_name=fold["split"],
                train_end=fold["train_end"],
                test_start=fold["test_start"],
                test_end=fold["test_end"],
                reserve_agg_full=reserve_agg_full,
                run_store_ids=selected_store_ids,
                min_train_days=min_train_days,
            )
            cv_summaries_local.append(fold_summary)
        cv_summary_local = pd.DataFrame(cv_summaries_local)

    rows_local = []
    pooled_chunks_local = []
    failures_local = []
    skipped_local = []

    for store_id in selected_store_ids:
        row, chunk, skip = evaluate_store_split(
            store_id,
            train_end=final_split_date,
            test_start=final_holdout_start,
            test_end=final_holdout_end,
            reserve_agg_full=reserve_agg_full,
            reserve_agg_capped=reserve_agg_final,
            min_train_days=min_train_days,
        )
        if skip is not None:
            skipped_local.append(skip)
            continue
        rows_local.append(row)
        pooled_chunks_local.append(chunk)
        for model_key in EXOG_CONFIGS:
            status_value = row.get(f"{model_key}_status")
            if isinstance(status_value, str) and "fallback_to_baseline" in status_value:
                failures_local.append({"air_store_id": store_id, "model": model_key, "error": status_value})

    store_metrics_local = pd.DataFrame(rows_local).merge(store_meta, on="air_store_id", how="left")
    pooled_df_local = pd.concat(pooled_chunks_local, ignore_index=True) if pooled_chunks_local else pd.DataFrame()
    failures_df_local = pd.DataFrame(failures_local)

    final_metrics_local = pooled_metric_table(pooled_df_local)
    metrics_lookup_local = final_metrics_local.set_index("model_key")
    summary_local = {
        "output_tag": output_tag,
        "n_stores_targeted": int(len(selected_store_ids)),
        "n_stores_evaluated": int(len(store_metrics_local)),
        "n_stores_skipped": int(len(skipped_local)),
        "n_model_fallbacks": int(len(failures_local)),
        "pooled_rows": int(len(pooled_df_local)),
        "seasonal_naive_rmsle_pooled": float(metrics_lookup_local.loc["seasonal_naive", "RMSLE"]),
        "sarima_no_exog_rmsle_pooled": float(metrics_lookup_local.loc["sarima_no_exog", "RMSLE"]),
        "arimax_holiday_only_rmsle_pooled": float(metrics_lookup_local.loc["arimax_holiday_only", "RMSLE"]),
        "sarima_better_than_naive_share": float((store_metrics_local["sarima_no_exog_rmsle"] < store_metrics_local["seasonal_naive_rmsle"]).mean()),
        "holiday_arimax_better_than_naive_share": float((store_metrics_local["arimax_holiday_only_rmsle"] < store_metrics_local["seasonal_naive_rmsle"]).mean()),
        "holiday_arimax_better_than_sarima_share": float((store_metrics_local["arimax_holiday_only_rmsle"] < store_metrics_local["sarima_no_exog_rmsle"]).mean()),
    }
    summary_local["sarima_improvement_vs_naive_pct"] = 100.0 * (
        summary_local["seasonal_naive_rmsle_pooled"] - summary_local["sarima_no_exog_rmsle_pooled"]
    ) / summary_local["seasonal_naive_rmsle_pooled"]
    summary_local["holiday_arimax_improvement_vs_naive_pct"] = 100.0 * (
        summary_local["seasonal_naive_rmsle_pooled"] - summary_local["arimax_holiday_only_rmsle_pooled"]
    ) / summary_local["seasonal_naive_rmsle_pooled"]
    summary_local["holiday_arimax_improvement_vs_sarima_pct"] = 100.0 * (
        summary_local["sarima_no_exog_rmsle_pooled"] - summary_local["arimax_holiday_only_rmsle_pooled"]
    ) / summary_local["sarima_no_exog_rmsle_pooled"]

    if not cv_summary_local.empty:
        cv_summary_local.to_csv(OUTPUT_DIR / f"store_level_cv_summary_{output_tag}.csv", index=False)
    store_metrics_local.to_csv(OUTPUT_DIR / f"store_level_metrics_{output_tag}.csv", index=False)
    pooled_df_local.to_csv(OUTPUT_DIR / f"store_level_holdout_predictions_{output_tag}.csv", index=False)
    final_metrics_local.to_csv(OUTPUT_DIR / f"store_level_ablation_metrics_{output_tag}.csv", index=False)
    with open(OUTPUT_DIR / f"store_level_summary_{output_tag}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **summary_local,
                "comparison_results": final_metrics_local[["model", "feature_set", "RMSLE", "RMSE_log", "MAPE"]].to_dict(orient="records"),
                "cv_folds": cv_summary_local.to_dict(orient="records") if not cv_summary_local.empty else [],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return summary_local, final_metrics_local, cv_summary_local, store_metrics_local

def representative_store_ids_for_diagnostics() -> list[str]:
    candidate_positions = sorted(set([0, 49, 124, 249, 399, len(eligible_store_ids) - 1]))
    ids = []
    for pos in candidate_positions:
        pos = min(max(pos, 0), len(eligible_store_ids) - 1)
        ids.append(eligible_store_ids.iloc[pos]["air_store_id"])
    return list(dict.fromkeys(ids))

def fit_store_residual_diagnostics(store_id: str) -> dict:
    store_df = build_store_daily_frame(store_id, visits, reserve_agg_full, reserve_agg_final, holidays)
    train = store_df.loc[store_df["visit_date"] <= final_split_date].copy()
    y_train = train["log1p_visitors"].astype(float)
    fit = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=50)
    resid = pd.Series(np.asarray(fit.resid, dtype=float)).dropna()
    lb = acorr_ljungbox(resid, lags=RESIDUAL_DIAGNOSTIC_LAGS, return_df=True)
    row = {
        "air_store_id": store_id,
        "train_days": int(len(train)),
        "aic": float(fit.aic),
        "resid_std": float(resid.std(ddof=1)),
    }
    for lag in RESIDUAL_DIAGNOSTIC_LAGS:
        row[f"ljung_box_pvalue_lag_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
        row[f"resid_acf_lag_{lag}"] = float(pd.Series(resid).autocorr(lag=lag))
    return row

if RUN_ROBUSTNESS_400:
    robustness_store_ids = eligible_store_ids.head(ROBUSTNESS_STORE_LIMIT)["air_store_id"].tolist()
    robustness_summary, robustness_metrics, robustness_cv_summary, robustness_store_metrics = run_panel_evaluation(
        selected_store_ids=robustness_store_ids,
        output_tag=ROBUSTNESS_OUTPUT_TAG,
        run_cv=False,
    )
    robustness_vs_main = pd.DataFrame([
        {
            "run": "main250cv",
            "stores_evaluated": summary["n_stores_evaluated"],
            "pooled_rows": summary["pooled_rows"],
            "naive_rmsle": summary["seasonal_naive_rmsle_pooled"],
            "sarima_rmsle": summary["sarima_no_exog_rmsle_pooled"],
            "holiday_arimax_rmsle": summary["arimax_holiday_only_rmsle_pooled"],
            "holiday_vs_sarima_pct": summary["holiday_arimax_improvement_vs_sarima_pct"],
        },
        {
            "run": ROBUSTNESS_OUTPUT_TAG,
            "stores_evaluated": robustness_summary["n_stores_evaluated"],
            "pooled_rows": robustness_summary["pooled_rows"],
            "naive_rmsle": robustness_summary["seasonal_naive_rmsle_pooled"],
            "sarima_rmsle": robustness_summary["sarima_no_exog_rmsle_pooled"],
            "holiday_arimax_rmsle": robustness_summary["arimax_holiday_only_rmsle_pooled"],
            "holiday_vs_sarima_pct": robustness_summary["holiday_arimax_improvement_vs_sarima_pct"],
        },
    ])
    display(robustness_vs_main)
else:
    print("400-store robustness check is disabled. Set RUN_ROBUSTNESS_400 = True to generate a deterministic robustness re-run.")

if RUN_REPRESENTATIVE_RESIDUAL_DIAGNOSTICS:
    representative_store_ids = representative_store_ids_for_diagnostics()
    residual_diagnostics_df = pd.DataFrame([fit_store_residual_diagnostics(store_id) for store_id in representative_store_ids])
    residual_diagnostics_df.to_csv(OUTPUT_DIR / f"{RESIDUAL_DIAGNOSTIC_OUTPUT_TAG}.csv", index=False)
    display(residual_diagnostics_df)

    fig, axes = plt.subplots(len(representative_store_ids), 1, figsize=(12, 3.5 * len(representative_store_ids)))
    if len(representative_store_ids) == 1:
        axes = [axes]
    for ax, store_id in zip(axes, representative_store_ids):
        store_df = build_store_daily_frame(store_id, visits, reserve_agg_full, reserve_agg_final, holidays)
        train = store_df.loc[store_df["visit_date"] <= final_split_date].copy()
        fit = SARIMAX(
            train["log1p_visitors"].astype(float),
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=50)
        sm.graphics.tsa.plot_acf(pd.Series(np.asarray(fit.resid, dtype=float)).dropna(), lags=28, ax=ax)
        ax.set_title(f"Representative store residual ACF: {store_id}")
    plt.tight_layout()
    plt.show()
else:
    print("Representative residual diagnostics are disabled. Set RUN_REPRESENTATIVE_RESIDUAL_DIAGNOSTICS = True to run them.")

# %%
print('Updated outputs saved in ../output')
print('Run mode       :', RUN_MODE)
print('Output tag     :', OUTPUT_TAG)
print('Model fallbacks:', len(failures))
print('Skipped stores :', len(skipped_stores))
if len(skipped_stores):
    display(skipped_df.head())
if len(failures):
    display(failures_df.head())

