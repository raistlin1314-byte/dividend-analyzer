#!/usr/bin/env python3
"""
险资红利策略 - Tushare版 (终极修复版 v2.0)

功能升级：
1. [智能回溯] 自动检测当日数据质量，如果市值缺失(NaN)，自动回退使用上一交易日数据。
2. [全量计算] 默认计算 200 只股票的 ERP 图表，不再只显示前 20 只。
3. [双重保险] 保留了"股价x股本"的手动计算逻辑，作为最后的兜底。
"""

import tushare as ts
import pandas as pd
import numpy as np
import json
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional
from math import erf, sqrt

class InsuranceStrategyTushare:
    """险资红利策略 - Tushare版"""
    
    def __init__(self,
                 token: str,
                 dividend_yield_threshold: float = 0.04,
                 market_cap_threshold: float = 100,
                 min_price: float = 5.0):
        self.dividend_yield_threshold = dividend_yield_threshold
        self.market_cap_threshold = market_cap_threshold
        self.min_price = min_price
        self.stats = defaultdict(int)
        
        # 初始化Tushare
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.connected = self._test_connection()
        
    def _test_connection(self) -> bool:
        """测试Tushare连接"""
        try:
            # 尝试获取上一年的最后一天，确保能拿到数据
            df = self.pro.trade_cal(exchange='SSE', start_date='20240101', end_date='20240101')
            if df is not None:
                print("✓ Tushare API 连接成功")
                return True
        except Exception as e:
            print(f"✗ Tushare API 连接失败: {e}")
        return False
    
    def _get_latest_trade_date(self) -> str:
        """获取最近交易日"""
        today = datetime.now()
        for i in range(10):
            date = (today - timedelta(days=i)).strftime('%Y%m%d')
            try:
                df = self.pro.trade_cal(exchange='SSE', start_date=date, end_date=date, is_open='1')
                if not df.empty:
                    return date
            except:
                continue
        return today.strftime('%Y%m%d')
    
    def get_stock_pool(self, use_sample: bool = False) -> pd.DataFrame:
        """获取全市场股票列表"""
        if use_sample:
            print("使用示例股票池")
            return pd.DataFrame({
                'ts_code': ['601398.SH', '601088.SH', '600900.SH', '002271.SZ', '600027.SH'],
                'name': ['工商银行', '中国神华', '长江电力', '东方雨虹', '华电国际']
            })
        
        try:
            print("获取全市场A股列表...")
            df = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )
            if df is None or df.empty:
                return self.get_stock_pool(use_sample=True)
            print(f"✓ 获取 {len(df)} 只股票")
            return df
        except Exception as e:
            print(f"✗ 获取股票列表异常: {e}")
            return self.get_stock_pool(use_sample=True)

    def get_daily_basic_data(self, trade_date: str = None) -> pd.DataFrame:
        """
        获取每日指标数据 (智能回溯版)
        逻辑：如果指定日期的数据存在大量空值(NaN)，则自动尝试前一天，直到找到有效数据。
        """
        if trade_date is None:
            trade_date = self._get_latest_trade_date()
        
        # 请求字段：增加了 total_share 以便手动计算市值
        req_fields = 'ts_code,trade_date,close,turnover_rate,pe_ttm,pb,ps_ttm,dv_ratio,dv_ttm,total_mv,circ_mv,total_share'
        
        print(f"准备获取每日指标数据 (起始日期: {trade_date})...")
        
        # 内部函数：尝试获取某一天的数据
        def fetch_one_day(date_str):
            try:
                df = self.pro.daily_basic(trade_date=date_str, fields=req_fields)
                if df is None or df.empty:
                    return None
                
                # 【关键逻辑】检查数据质量
                # 如果总市值(total_mv)缺失率超过 50%，说明 Tushare 还没更新完数据，该日数据无效
                nan_ratio = df['total_mv'].isna().mean()
                if nan_ratio > 0.5:
                    print(f"  ⚠ {date_str} 数据质量差 (市值缺失率 {nan_ratio:.1%})，跳过...")
                    return None
                return df
            except:
                return None

        # 1. 优先尝试指定日期
        df = fetch_one_day(trade_date)
        
        # 2. 如果数据无效，回溯最近 5 个交易日
        if df is None:
            print(f"  ⚠ 正在回溯寻找最近的完整数据...")
            current_dt = datetime.strptime(trade_date, '%Y%m%d')
            for i in range(1, 6): # 最多回溯5天
                prev_date = (current_dt - timedelta(days=i)).strftime('%Y%m%d')
                df = fetch_one_day(prev_date)
                if df is not None:
                    print(f"  ✓ 成功回溯，使用 {prev_date} 的完整数据")
                    break
        
        if df is not None:
            print(f"✓ 获取 {len(df)} 条有效指标数据")
            return df
        else:
            print("✗ 无法获取有效的每日指标数据 (已尝试回溯)")
            return pd.DataFrame()
    
    def screen_stocks(self, stock_pool: pd.DataFrame) -> pd.DataFrame:
        """执行筛选"""
        if not self.connected:
            return pd.DataFrame()
        
        print(f"\n{'='*60}")
        print("开始筛选高股息股票")
        print("="*60)
        
        # 获取智能回溯后的每日数据
        daily_data = self.get_daily_basic_data()
        if daily_data.empty:
            return pd.DataFrame()
        
        # 合并数据
        df = pd.merge(stock_pool, daily_data, on='ts_code', how='inner')
        
        # 【双重保险】修复市值：如果 Tushare 给的 total_mv 还是空，用 股价*股本 计算
        def fix_market_cap(row):
            if pd.notna(row['total_mv']):
                return row['total_mv']
            if pd.notna(row['total_share']) and pd.notna(row['close']):
                return row['total_share'] * row['close']
            return np.nan

        if 'total_share' in df.columns:
            df['total_mv'] = df.apply(fix_market_cap, axis=1)
        
        # 1. 排除ST
        df = df[~df['name'].str.contains('ST|\\*ST', case=False, na=False)]
        
        # 2. 补全股息率
        df['dividend_yield'] = df['dv_ttm'].fillna(df['dv_ratio'])
        
        # 3. 筛选有效数据
        df_clean = df.dropna(subset=['dividend_yield', 'total_mv', 'close'])
        
        # 4. 执行阈值筛选
        df_clean['dividend_yield_pct'] = df_clean['dividend_yield'] / 100
        
        # 股息率筛选
        df_div = df_clean[df_clean['dividend_yield_pct'] >= self.dividend_yield_threshold]
        
        # 市值筛选 (万元 -> 亿元)
        df_div['mkt_cap_yi'] = df_div['total_mv'] / 10000
        df_cap = df_div[df_div['mkt_cap_yi'] >= self.market_cap_threshold]
        
        # 股价筛选
        df_final = df_cap[df_cap['close'] >= self.min_price]
        
        # 整理输出列
        result = df_final[[
            'ts_code', 'name', 'industry', 'close',
            'dividend_yield_pct', 'mkt_cap_yi', 'pe_ttm', 'pb'
        ]].copy()
        
        result.columns = [
            'ts_code', 'name', 'industry', 'close',
            'dividend_yield', 'market_cap', 'pe_ttm', 'pb'
        ]
        
        result = result.sort_values('dividend_yield', ascending=False)
        result = result.reset_index(drop=True)
        
        print(f"✓ 筛选完成，共 {len(result)} 只股票符合条件")
        return result

    def print_results(self, result_df: pd.DataFrame, top_n: int = 30):
        if result_df.empty:
            print("未找到符合条件的股票")
            return
        
        print(f"\n🏆 Top {min(top_n, len(result_df))} 高股息股票:")
        print("-"*60)
        display_df = result_df.head(top_n)
        for idx, row in display_df.iterrows():
            print(f"{idx+1:3d}. {row['name']:8s} ({row['ts_code']}) | "
                  f"股息: {row['dividend_yield']*100:5.2f}% | "
                  f"市值: {row['market_cap']:8.1f}亿")


class ERPCalculator:
    """ERP计算器 (包含重试机制)"""
    
    def __init__(self, token: str):
        ts.set_token(token)
        self.pro = ts.pro_api()
    
    def get_dividend_yield_history(self, ts_code: str, years: int = 12) -> pd.DataFrame:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y%m%d')
        
        try:
            df = self.pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,dv_ratio,dv_ttm,close,pe_ttm'
            )
            if df is None or df.empty:
                return pd.DataFrame()
            
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            df = df.drop_duplicates(subset=['trade_date'], keep='last')
            
            # 填充缺失的股息率
            df['dividend_yield'] = df['dv_ttm'].fillna(df['dv_ratio']).fillna(0)
            
            return df[['trade_date', 'dividend_yield', 'close', 'pe_ttm']]
        except:
            return pd.DataFrame()
    
    def get_risk_free_rate(self, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            df = self.pro.shibor(start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['date'])
                df = df.drop_duplicates(subset=['trade_date'], keep='last')
                df['risk_free_rate'] = df['1y'] / 100
                return df[['trade_date', 'risk_free_rate']].dropna()
        except:
            pass
        
        # 失败则使用固定利率兜底
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.DataFrame({'trade_date': dates, 'risk_free_rate': 0.02})
    
    def calculate_erp(self, ts_code: str, years: int = 12) -> Dict:
        div_df = self.get_dividend_yield_history(ts_code, years)
        if div_df.empty or len(div_df) < 252:
            return {}
        
        start_date = div_df['trade_date'].min().strftime('%Y%m%d')
        end_date = div_df['trade_date'].max().strftime('%Y%m%d')
        rf_df = self.get_risk_free_rate(start_date, end_date)
        
        div_df = div_df.set_index('trade_date')
        rf_df = rf_df.set_index('trade_date')
        rf_df = rf_df[~rf_df.index.duplicated(keep='last')]
        rf_df = rf_df.reindex(div_df.index, method='ffill')
        
        merged = div_df.join(rf_df, how='left')
        merged['risk_free_rate'] = merged['risk_free_rate'].ffill().fillna(0.02)
        merged['erp'] = merged['dividend_yield'] - merged['risk_free_rate'] * 100
        merged = merged.reset_index()
        
        rolling_stats = self._calculate_rolling_stats(merged)
        
        # 采样压缩 (每 2 天取一个点，减小JSON体积)
        step = 2 
        
        current_erp = merged['erp'].iloc[-1]
        
        result = {
            'dates': merged['trade_date'].dt.strftime('%Y-%m-%d').tolist()[::step],
            'erp': [round(x, 2) for x in merged['erp'].tolist()[::step]],
            'mean': [round(x, 2) for x in rolling_stats['5y']['mean'].tolist()[::step]], # 默认显示5y均值
            'rolling_stats': {},
            'current_position': {'erp': round(float(current_erp), 4)}
        }
        
        # 将复杂的统计数据放入 rolling_stats
        for period, stats in rolling_stats.items():
            result['rolling_stats'][period] = {
                'mean': [round(x, 2) if pd.notna(x) else None for x in stats['mean'].tolist()[::step]],
                'p1std': [round(x, 2) if pd.notna(x) else None for x in stats['+1std'].tolist()[::step]],
                'p2std': [round(x, 2) if pd.notna(x) else None for x in stats['+2std'].tolist()[::step]],
                'm1std': [round(x, 2) if pd.notna(x) else None for x in stats['-1std'].tolist()[::step]],
                'm2std': [round(x, 2) if pd.notna(x) else None for x in stats['-2std'].tolist()[::step]],
            }
        
        return result
    
    def _calculate_rolling_stats(self, df: pd.DataFrame) -> Dict:
        results = {}
        windows = {'3y': 756, '5y': 1260, '10y': 2520}
        for period, window_days in windows.items():
            if len(df) < window_days // 2: # 稍微放宽限制
                continue
            rolling_mean = df['erp'].rolling(window=window_days, min_periods=window_days//2).mean()
            rolling_std = df['erp'].rolling(window=window_days, min_periods=window_days//2).std()
            results[period] = {
                'mean': rolling_mean,
                '+1std': rolling_mean + rolling_std,
                '+2std': rolling_mean + 2 * rolling_std,
                '-1std': rolling_mean - rolling_std,
                '-2std': rolling_mean - 2 * rolling_std,
            }
        return results

def save_results(stock_pool: pd.DataFrame, erp_data: Dict, output_dir: str, params: Dict):
    os.makedirs(output_dir, exist_ok=True)
    
    stocks = stock_pool.to_dict(orient='records')
    for s in stocks:
        for k, v in s.items():
            if isinstance(v, float):
                s[k] = round(v, 4)
                
    stock_pool_file = os.path.join(output_dir, 'stock_pool.json')
    with open(stock_pool_file, 'w', encoding='utf-8') as f:
        json.dump({
            'updated_at': datetime.now().isoformat(),
            'screening_params': params,
            'total_count': len(stocks),
            'stocks': stocks
        }, f, ensure_ascii=False, indent=2)
    
    if erp_data:
        erp_file = os.path.join(output_dir, 'erp_data.json')
        with open(erp_file, 'w', encoding='utf-8') as f:
            json.dump({'updated_at': datetime.now().isoformat(), 'stocks': erp_data}, f, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='险资红利策略')
    parser.add_argument('--token', required=True)
    parser.add_argument('--mode', type=int, default=2)
    parser.add_argument('--min-yield', type=float, default=4)
    parser.add_argument('--min-cap', type=float, default=100)
    parser.add_argument('--min-price', type=float, default=5)
    parser.add_argument('--output', default='./data')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--no-erp', action='store_true')
    
    # 【修改点】默认值从 20 改为 200，保证所有筛选出来的股票都计算图表
    parser.add_argument('--erp-count', type=int, default=200)
    
    args = parser.parse_args()
    
    # 模式设置
    if args.mode == 1:
        div_yield, mkt_cap, min_price = 0.03, 50, 3.0
        mode_name = "宽松模式"
    elif args.mode == 2:
        div_yield, mkt_cap, min_price = 0.04, 100, 5.0
        mode_name = "标准模式"
    elif args.mode == 3:
        div_yield, mkt_cap, min_price = 0.05, 200, 5.0
        mode_name = "严格模式"
    else:
        div_yield = args.min_yield / 100
        mkt_cap = args.min_cap
        min_price = args.min_price
        mode_name = "自定义模式"
    
    print("="*60)
    print(f"策略执行: {mode_name} | 计算数量: {args.erp_count}")
    print("="*60)
    
    strategy = InsuranceStrategyTushare(args.token, div_yield, mkt_cap, min_price)
    
    if not strategy.connected:
        sys.exit(1)
        
    stock_pool = strategy.get_stock_pool(use_sample=args.sample)
    if stock_pool.empty:
        sys.exit(1)
        
    results = strategy.screen_stocks(stock_pool)
    strategy.print_results(results, top_n=20)
    
    if results.empty:
        sys.exit(0)
    
    # 计算ERP
    erp_data = {}
    if not args.no_erp and len(results) > 0:
        print(f"\n开始计算 ERP 图表数据 (目标: {min(args.erp_count, len(results))} 只)...")
        erp_calculator = ERPCalculator(args.token)
        
        # 限制计算数量
        calc_list = results.head(args.erp_count)
        
        for idx, row in calc_list.iterrows():
            ts_code = row['ts_code']
            
            # 【重试机制】
            for attempt in range(3):
                try:
                    print(f"[{idx+1}/{len(calc_list)}] 计算 {row['name']}...", end="\r")
                    erp_result = erp_calculator.calculate_erp(ts_code)
                    if erp_result:
                        erp_data[ts_code] = {
                            'name': row['name'],
                            'industry': row['industry'] if pd.notna(row['industry']) else '',
                            **erp_result
                        }
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        print(f"\n  ✗ {row['name']} 计算失败")
            
            time.sleep(0.1) # 礼貌等待
            
    print(f"\n计算完成，成功生成 {len(erp_data)} 只股票图表数据")
    
    save_results(results, erp_data, args.output, {
        'mode': mode_name,
        'min_dividend_yield': div_yield,
        'min_market_cap': mkt_cap,
        'min_price': min_price
    })
    print("\n✓ 任务全部完成")

if __name__ == '__main__':
    main()