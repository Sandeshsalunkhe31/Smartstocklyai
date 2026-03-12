from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="StockSense Prophet API",
    description="AI-powered inventory forecasting",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SalesDataPoint(BaseModel):
    date: str
    units: float

class ForecastRequest(BaseModel):
    sales_data: List[SalesDataPoint]
    periods: int = 30
    current_inventory: Optional[float] = None
    lead_time_days: int = 14

class ForecastResponse(BaseModel):
    forecast_value: float
    daily_forecast: float
    confidence_score: float
    confidence_interval_low: float
    confidence_interval_high: float
    trend_analysis: str
    seasonality_detected: bool
    data_quality_score: float
    recommendation_text: str
    stockout_risk: str
    stockout_date: Optional[str] = None
    reorder_quantity: Optional[float] = None
    reorder_point: Optional[float] = None

def calculate_data_quality(df):
    score = 100.0
    if len(df) < 30:
        score -= (30 - len(df)) * 2
    date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
    missing_dates = len(date_range) - len(df)
    score -= missing_dates * 3
    cv = df['y'].std() / df['y'].mean() if df['y'].mean() > 0 else 0
    if cv > 1.0:
        score -= min(20, (cv - 1.0) * 10)
    zero_ratio = (df['y'] == 0).sum() / len(df)
    score -= zero_ratio * 30
    return max(0, min(100, score))

def detect_seasonality(df):
    if len(df) < 14:
        return False
    df['day_of_week'] = df['ds'].dt.dayofweek
    weekly_avg = df.groupby('day_of_week')['y'].mean()
    cv = weekly_avg.std() / weekly_avg.mean() if weekly_avg.mean() > 0 else 0
    return cv > 0.3

def calculate_trend(df):
    if len(df) < 7:
        return "Insufficient data"
    x = np.arange(len(df))
    y = df['y'].values
    mask = y > 0
    if mask.sum() < 3:
        return "Stable"
    x_clean = x[mask]
    y_clean = y[mask]
    slope = np.polyfit(x_clean, y_clean, 1)[0]
    mean_y = y_clean.mean()
    trend_pct = (slope * len(df) / mean_y * 100) if mean_y > 0 else 0
    if abs(trend_pct) < 5:
        return "Stable"
    elif trend_pct > 20:
        return f"Strong upward trend (+{trend_pct:.0f}%)"
    elif trend_pct > 5:
        return f"Upward trend (+{trend_pct:.0f}%)"
    elif trend_pct < -20:
        return f"Strong downward trend ({trend_pct:.0f}%)"
    else:
        return f"Downward trend ({trend_pct:.0f}%)"

def simple_forecast(df, periods):
    if len(df) >= 28:
        recent_avg = df.tail(14)['y'].mean()
        prev_avg = df.iloc[-28:-14]['y'].mean()
        trend_factor = recent_avg / prev_avg if prev_avg > 0 else 1.0
    else:
        trend_factor = 1.0
    weights = np.exp(np.linspace(-1, 0, min(30, len(df))))
    weights = weights / weights.sum()
    recent_data = df.tail(min(30, len(df)))['y'].values
    weighted_avg = (recent_data * weights).sum()
    daily_forecast = weighted_avg * trend_factor
    total_forecast = daily_forecast * periods
    cv = df['y'].std() / df['y'].mean() if df['y'].mean() > 0 else 1.0
    confidence = max(60, 90 - cv * 30)
    ci_low = total_forecast * 0.85
    ci_high = total_forecast * 1.15
    return {
        'forecast': total_forecast,
        'daily_forecast': daily_forecast,
        'confidence': confidence,
        'yhat_lower': ci_low,
        'yhat_upper': ci_high
    }

def prophet_forecast(df, periods):
    try:
        from prophet import Prophet
        model = Prophet(
            daily_seasonality=True if len(df) >= 14 else False,
            weekly_seasonality=True if len(df) >= 14 else False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.80
        )
        import logging as prophet_logging
        prophet_logging.getLogger('prophet').setLevel(prophet_logging.WARNING)
        model.fit(df)
        future = model.make_future_dataframe(periods=periods)
        forecast_df = model.predict(future)
        forecast_future = forecast_df.tail(periods)
        total_forecast = forecast_future['yhat'].sum()
        daily_forecast = forecast_future['yhat'].mean()
        ci_low = forecast_future['yhat_lower'].sum()
        ci_high = forecast_future['yhat_upper'].sum()
        ci_width = ci_high - ci_low
        relative_width = ci_width / total_forecast if total_forecast > 0 else 1.0
        confidence = max(70, min(95, 95 - relative_width * 50))
        return {
            'forecast': max(0, total_forecast),
            'daily_forecast': max(0, daily_forecast),
            'confidence': confidence,
            'yhat_lower': max(0, ci_low),
            'yhat_upper': max(0, ci_high)
        }
    except:
        return simple_forecast(df, periods)

def calculate_stockout_info(current_inventory, daily_demand, lead_time):
    if current_inventory <= 0 or daily_demand <= 0:
        return {
            'risk': 'critical',
            'stockout_date': datetime.now().strftime('%Y-%m-%d'),
            'reorder_quantity': daily_demand * (lead_time + 7),
            'reorder_point': daily_demand * lead_time * 1.5
        }
    days_remaining = current_inventory / daily_demand
    reorder_point = daily_demand * lead_time * 1.5
    reorder_quantity = daily_demand * (lead_time + 14)
    if days_remaining < lead_time:
        risk = 'critical'
    elif days_remaining < lead_time * 1.5:
        risk = 'high'
    elif days_remaining < lead_time * 2:
        risk = 'medium'
    else:
        risk = 'low'
    stockout_date = (datetime.now() + timedelta(days=int(days_remaining))).strftime('%Y-%m-%d')
    return {
        'risk': risk,
        'stockout_date': stockout_date,
        'reorder_quantity': round(reorder_quantity, 0),
        'reorder_point': round(reorder_point, 0)
    }

@app.get("/")
async def root():
    try:
        import prophet
        prophet_available = True
    except:
        prophet_available = False
    return {
        "status": "healthy",
        "version": "1.0.0",
        "prophet_available": prophet_available
    }

@app.post("/forecast", response_model=ForecastResponse)
async def create_forecast(request: ForecastRequest):
    try:
        data = []
        for point in request.sales_data:
            try:
                date = pd.to_datetime(point.date)
                data.append({'ds': date, 'y': float(point.units)})
            except:
                raise HTTPException(status_code=400, detail=f"Invalid date format: {point.date}")
        df = pd.DataFrame(data)
        df = df.sort_values('ds')
        if len(df) < 7:
            raise HTTPException(status_code=400, detail="At least 7 days of data required")
        data_quality = calculate_data_quality(df)
        has_seasonality = detect_seasonality(df)
        trend = calculate_trend(df)
        forecast_result = prophet_forecast(df, request.periods)
        if forecast_result['forecast'] > df['y'].tail(request.periods).sum() * 1.2:
            recommendation = "Demand increasing. Consider ordering more inventory."
        elif forecast_result['forecast'] < df['y'].tail(request.periods).sum() * 0.8:
            recommendation = "Demand decreasing. Reduce order quantities."
        else:
            recommendation = "Demand stable. Maintain current inventory levels."
        stockout_info = None
        if request.current_inventory is not None and request.current_inventory >= 0:
            stockout_info = calculate_stockout_info(
                request.current_inventory,
                forecast_result['daily_forecast'],
                request.lead_time_days
            )
            if stockout_info['risk'] == 'critical':
                recommendation = f"URGENT: Reorder {stockout_info['reorder_quantity']:.0f} units immediately!"
            elif stockout_info['risk'] == 'high':
                recommendation = f"Reorder soon: {stockout_info['reorder_quantity']:.0f} units recommended"
        response = ForecastResponse(
            forecast_value=round(forecast_result['forecast'], 2),
            daily_forecast=round(forecast_result['daily_forecast'], 2),
            confidence_score=round(forecast_result['confidence'], 1),
            confidence_interval_low=round(forecast_result['yhat_lower'], 2),
            confidence_interval_high=round(forecast_result['yhat_upper'], 2),
            trend_analysis=trend,
            seasonality_detected=has_seasonality,
            data_quality_score=round(data_quality, 1),
            recommendation_text=recommendation,
            stockout_risk=stockout_info['risk'] if stockout_info else 'unknown',
            stockout_date=stockout_info['stockout_date'] if stockout_info else None,
            reorder_quantity=stockout_info['reorder_quantity'] if stockout_info else None,
            reorder_point=stockout_info['reorder_point'] if stockout_info else None
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        import prophet
        prophet_status = "available"
    except:
        prophet_status = "not installed"
    return {
        "status": "healthy",
        "prophet": prophet_status,
        "timestamp": datetime.now().isoformat()
    }
