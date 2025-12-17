#!/usr/bin/env python3
"""
险资红利策略 - Tushare版 (增强修复版)
基于原Wind版本改写，适配Tushare数据库

功能：
1. 筛选高股息红利股（股息率、市值、股价）
2. 生成ERP滚动曲线数据
3. 输出JSON文件供前端使用

修复日志：
- 修复 Tushare 市值数据偶尔缺失(NaN)的问题 (通过总股本手动计算)
- 增加 网络请求重试机制，解决老牌股票因网络波动无数据的问题
- 增加 API请求间隔，防止触发频率限制
"""

import tushare as ts
import pandas as pd
import numpy as np
import json
import argparse
import os
import sys
import time  # 确保导入 time 模块
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from math import erf, sqrt


class InsuranceStrategyTushare:
    """险资红利策略 - Tushare版"""
    
    def __init__(self,
                 token: str,
                 dividend_yield_threshold: float = 0.04,
                 market_cap_threshold: float = 100,
                 min_price: float = 5.0):
        """
        初始化策略
        """
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
            df = self.pro.trade_cal(exchange='SSE', start_date='20240101', end_date='20240101')
            if df is not None and not df.empty:
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
                df = self.pro.trade_cal(
                    exchange='SSE',
                    start_date=date,
                    end_date=date,
                    is_open='1'
                )
                if not df.empty:
                    return date
            except:
                continue
        return today.strftime('%Y%m%d')
    
    def get_stock_pool(self, use_sample: bool = False) -> pd.DataFrame:
        """获取股票池"""
        if use_sample:
            print("使用示例股票池 (30只)")
            return pd.DataFrame({
                'ts_code': [
                    '601398.SH', '601988.SH', '600036.SH', '601288.SH',
                    '600000.SH', '601166.SH', '600015.SH', '601939.SH',
                    '601857.SH', '600028.SH', '600688.SH',
                    '600900.SH', '600886.SH', '600795.SH',
                    '601088.SH', '601225.SH', '600188.SH',
                    '601318.SH', '601628.SH', '601601.SH',
                    '600019.SH', '600309.SH', '601919.SH',
                    '600585.SH', '601898.SH', '600011.SH',
                    '601006.SH', '600031.SH', '600009.SH', '601991.SH'
                ],
                'name': [
                    '工商银行', '中国银行', '招商银行', '农业银行',
                    '浦发银行', '兴业银行', '华夏银行', '建设银行',
                    '中国石油', '中国石化', '上海石化',
                    '长江电力', '国投电力', '国电电力',
                    '中国神华', '陕西煤业', '兖矿能源',
                    '中国平安', '中国人寿', '中国太保',
                    '宝钢股份', '万华化学', '中远海控',
                    '海螺水泥', '中煤能源', '华能国际',
                    '大秦铁路', '三一重工', '上海机场', '大唐发电'
                ]
            })
        
        try:
            print("获取全市场A股列表...")
            df = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )
            
            if df is None or df.empty:
                print("⚠ 获取全市场失败，使用示例股票池")
                return self.get_stock_pool(use_sample=True)
            
            print(f"✓ 获取 {len(df)} 只股票")
            return df
            
        except Exception as e:
            print(f"✗ 获取股票列表异常: {e}")
            return self.get_stock_pool(use_sample=True)
    
    def get_daily_basic_data(self, trade_date: str = None) -> pd.DataFrame:
        """
        获取每日指标数据
        【修复】增加了 total_share 字段，用于在 total_mv 缺失时手动计算市值
        """
        if trade_date is None:
            trade_date = self._get_latest_trade_date()
        
        print(f"获取 {trade_date} 的每日指标数据...")
        
        # 增加 total_share 字段
        req_fields = 'ts_code,trade_date,close,turnover_rate,pe_ttm,pb,ps_ttm,dv_ratio,dv_ttm,total_mv,circ_mv,total_share'
        
        try:
            df = self.pro.daily_basic(trade_date=trade_date, fields=req_fields)
            
            if df is None or df.empty:
                print(f"⚠ {trade_date} 无数据，尝试前一交易日")
                for i in range(1, 5):
                    prev_date = (datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=i)).strftime('%Y%m%d')
                    df = self.pro.daily_basic(trade_date=prev_date, fields=req_fields)
                    if df is not None and not df.empty:
                        print(f"✓ 使用 {prev_date} 数据")
                        break
            
            if df is not None and not df.empty:
                print(f"✓ 获取 {len(df)} 条每日指标数据")
                return df
            else:
                print("✗ 无法获取每日指标数据")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"✗ 获取每日指标异常: {e}")
            return pd.DataFrame()
    
    def screen_stocks(self, stock_pool: pd.DataFrame) -> pd.DataFrame:
        """执行股票筛选"""
        if not self.connected:
            print("✗ Tushare未连接")
            return pd.DataFrame()
        
        print(f"\n{'='*60}")
        print("开始筛选高股息股票")
        print("="*60)
        print(f"筛选条件:")
        print(f"  • 股息率 ≥ {self.dividend_yield_threshold*100}%")
        print(f"  • 市值 ≥ {self.market_cap_threshold} 亿元")
        print(f"  • 股价 ≥ {self.min_price} 元")
        print("-"*60)
        
        self.stats = defaultdict(int)
        self.stats['总数'] = len(stock_pool)
        
        daily_data = self.get_daily_basic_data()
        if daily_data.empty:
            print("✗ 无法获取每日指标数据")
            return pd.DataFrame()
        
        df = pd.merge(stock_pool, daily_data, on='ts_code', how='inner')
        print(f"合并后有效数据: {len(df)} 条")
        
        # 【修复】市值数据填充逻辑
        # Tushare中: total_mv单位是万元, total_share单位是万股, close是元
        # 所以 total_share * close 直接等于 total_mv (万元)
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
        self.stats['排除ST后'] = len(df)
        print(f"✓ 排除ST后: {len(df)} 只")
        
        # 2. 数据完整性检查
        df['dividend_yield'] = df['dv_ttm'].fillna(df['dv_ratio'])
        df_clean = df.dropna(subset=['dividend_yield', 'total_mv', 'close'])
        self.stats['数据完整'] = len(df_clean)
        print(f"✓ 数据完整: {len(df_clean)} 只")
        
        # 3. 股息率筛选
        df_clean['dividend_yield_pct'] = df_clean['dividend_yield'] / 100
        df_div = df_clean[df_clean['dividend_yield_pct'] >= self.dividend_yield_threshold]
        self.stats['股息率达标'] = len(df_div)
        print(f"✓ 股息率≥{self.dividend_yield_threshold*100}%: {len(df_div)} 只")
        
        # 4. 市值筛选 (万元 -> 亿元)
        df_div['mkt_cap_yi'] = df_div['total_mv'] / 10000
        df_cap = df_div[df_div['mkt_cap_yi'] >= self.market_cap_threshold]
        self.stats['市值达标'] = len(df_cap)
        print(f"✓ 市值≥{self.market_cap_threshold}亿: {len(df_cap)} 只")
        
        # 5. 股价筛选
        df_final = df_cap[df_cap['close'] >= self.min_price]
        self.stats['价格达标'] = len(df_final)
        print(f"✓ 股价≥{self.min_price}元: {len(df_final)} 只")
        
        # 整理结果
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
        
        print("-"*60)
        print(f"✓ 筛选完成，共 {len(result)} 只股票符合条件")
        return result
    
    def print_results(self, result_df: pd.DataFrame, top_n: int = 50):
        """打印筛选结果"""
        print("\n" + "="*60)
        print("筛选结果".center(60))
        print("="*60)
        
        if result_df.empty:
            print("\n❌ 未找到符合条件的股票")
            return
        
        print(f"\n✓ 找到 {len(result_df)} 只符合条件的股票")
        print(f"  平均股息率: {result_df['dividend_yield'].mean()*100:.2f}%")
        print(f"  平均市值: {result_df['market_cap'].mean():.1f}亿元")
        
        display_df = result_df.head(top_n)
        print(f"\n🏆 Top {min(top_n, len(result_df))} 高股息股票:")
        print("-"*60)
        
        for idx, row in display_df.iterrows():
            rank = idx + 1
            print(f"{rank:3d}. {row['name']:8s} ({row['ts_code']})")
            print(f"     股息率: {row['dividend_yield']*100:5.2f}% | "
                  f"股价: ¥{row['close']:7.2f} | "
                  f"市值: {row['market_cap']:8.1f}亿")
            print()


class ERPCalculator:
    """ERP计算器"""
    
    def __init__(self, token: str):
        ts.set_token(token)
        self.pro = ts.pro_api()
    
    def get_dividend_yield_history(self, ts_code: str, years: int = 12) -> pd.DataFrame:
        """获取股息率历史数据"""
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
            
            df['dividend_yield'] = df['dv_ttm'].fillna(df['dv_ratio'])
            # 【修复】将空的分红数据填充为0，防止计算中断
            df['dividend_yield'] = df['dividend_yield'].fillna(0)
            
            return df[['trade_date', 'dividend_yield', 'close', 'pe_ttm']]
            
        except Exception as e:
            print(f"  ✗ 获取历史数据失败: {e}")
            return pd.DataFrame()
    
    def get_risk_free_rate(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取无风险利率（SHIBOR 1年期）"""
        try:
            df = self.pro.shibor(start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['date'])
                df = df.drop_duplicates(subset=['trade_date'], keep='last')
                df['risk_free_rate'] = df['1y'] / 100
                return df[['trade_date', 'risk_free_rate']].dropna()
        except:
            pass
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.DataFrame({
            'trade_date': dates,
            'risk_free_rate': 0.02
        })
    
    def calculate_erp(self, ts_code: str, years: int = 12) -> Dict:
        """计算ERP滚动曲线数据"""
        div_df = self.get_dividend_yield_history(ts_code, years)
        if div_df.empty or len(div_df) < 252:
            return {}
        
        start_date = div_df['trade_date'].min().strftime('%Y%m%d')
        end_date = div_df['trade_date'].max().strftime('%Y%m%d')
        rf_df = self.get_risk_free_rate(start_date, end_date)
        
        div_df = div_df.set_index('trade_date')
        rf_df = rf_df.drop_duplicates(subset=['trade_date'], keep='last')
        rf_df = rf_df.set_index('trade_date')
        rf_df = rf_df[~rf_df.index.duplicated(keep='last')]
        rf_df = rf_df.reindex(div_df.index, method='ffill')
        
        merged = div_df.join(rf_df, how='left')
        merged['risk_free_rate'] = merged['risk_free_rate'].ffill().fillna(0.02)
        
        merged['erp'] = merged['dividend_yield'] - merged['risk_free_rate'] * 100
        merged = merged.reset_index()
        
        rolling_stats = self._calculate_rolling_stats(merged)
        
        # 计算当前位置Z-Score
        current_erp = merged['erp'].iloc[-1]
        current_position = {}
        
        for period, stats in rolling_stats.items():
            if stats['mean'].iloc[-1] is not None and not pd.isna(stats['mean'].iloc[-1]):
                latest_mean = stats['mean'].iloc[-1]
                latest_std = stats['std'].iloc[-1]
                
                if latest_std > 0:
                    z_score = (current_erp - latest_mean) / latest_std
                    current_position[period] = {
                        'erp': round(float(current_erp), 4),
                        'mean': round(float(latest_mean), 4),
                        'std': round(float(latest_std), 4),
                        'z_score': round(float(z_score), 4),
                        'percentile': round(self._z_to_percentile(z_score), 2)
                    }
        
        # 采样压缩数据
        step = max(1, len(merged) // 300)
        
        result = {
            'dates': merged['trade_date'].dt.strftime('%Y-%m-%d').tolist()[::step],
            'erp': [round(x, 2) for x in merged['erp'].tolist()[::step]],
            'dividend_yield': [round(x, 2) for x in merged['dividend_yield'].tolist()[::step]],
            'close': [round(x, 2) for x in merged['close'].tolist()[::step]],
            'rolling_stats': {},
            'current_position': current_position
        }
        
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
        """计算滚动统计"""
        results = {}
        windows = {'3y': 756, '5y': 1260, '10y': 2520}
        
        for period, window_days in windows.items():
            if len(df) < window_days // 2:
                continue
            
            rolling_mean = df['erp'].rolling(window=window_days, min_periods=window_days//2).mean()
            rolling_std = df['erp'].rolling(window=window_days, min_periods=window_days//2).std()
            
            results[period] = {
                'mean': rolling_mean,
                'std': rolling_std,
                '+1std': rolling_mean + rolling_std,
                '+2std': rolling_mean + 2 * rolling_std,
                '-1std': rolling_mean - rolling_std,
                '-2std': rolling_mean - 2 * rolling_std,
            }
        return results
    
    def _z_to_percentile(self, z: float) -> float:
        return (1 + erf(z / sqrt(2))) / 2 * 100


def save_results(stock_pool: pd.DataFrame, erp_data: Dict, output_dir: str, params: Dict):
    """保存结果为JSON文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    stocks = stock_pool.to_dict(orient='records')
    for s in stocks:
        s['dividend_yield'] = round(s['dividend_yield'], 4)
        s['market_cap'] = round(s['market_cap'], 2)
        s['close'] = round(s['close'], 2)
        s['pe_ttm'] = round(s['pe_ttm'], 2) if pd.notna(s.get('pe_ttm')) else None
        s['pb'] = round(s['pb'], 2) if pd.notna(s.get('pb')) else None
    
    stock_pool_file = os.path.join(output_dir, 'stock_pool.json')
    with open(stock_pool_file, 'w', encoding='utf-8') as f:
        json.dump({
            'updated_at': datetime.now().isoformat(),
            'screening_params': params,
            'total_count': len(stocks),
            'stocks': stocks
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 股票池已保存: {stock_pool_file}")
    
    if erp_data:
        erp_file = os.path.join(output_dir, 'erp_data.json')
        with open(erp_file, 'w', encoding='utf-8') as f:
            json.dump({
                'updated_at': datetime.now().isoformat(),
                'stocks': erp_data
            }, f, ensure_ascii=False)
        print(f"✓ ERP数据已保存: {erp_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='险资红利策略 - Tushare版')
    parser.add_argument('--token', required=True, help='Tushare API Token')
    parser.add_argument('--mode', type=int, default=2, choices=[1,2,3,4], help='模式')
    parser.add_argument('--min-yield', type=float, default=4, help='自定义股息率')
    parser.add_argument('--min-cap', type=float, default=100, help='自定义市值')
    parser.add_argument('--min-price', type=float, default=5, help='自定义股价')
    parser.add_argument('--output', default='./data', help='输出目录')
    parser.add_argument('--sample', action='store_true', help='使用样本股票池')
    parser.add_argument('--no-erp', action='store_true', help='不计算ERP')
    parser.add_argument('--erp-count', type=int, default=20, help='计算ERP数量')
    
    args = parser.parse_args()
    
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
    print(f"险资红利策略 - {mode_name}")
    print("="*60)
    
    strategy = InsuranceStrategyTushare(args.token, div_yield, mkt_cap, min_price)
    
    if not strategy.connected:
        sys.exit(1)
    
    stock_pool = strategy.get_stock_pool(use_sample=args.sample)
    if stock_pool.empty:
        sys.exit(1)
    
    results = strategy.screen_stocks(stock_pool)
    strategy.print_results(results, top_n=30)
    
    if results.empty:
        sys.exit(0)
    
    erp_data = {}
    if not args.no_erp and len(results) > 0:
        print(f"\n{'='*60}")
        print(f"计算ERP数据 (前{min(args.erp_count, len(results))}只股票)")
        print("="*60)
        
        erp_calculator = ERPCalculator(args.token)
        
        for idx, row in results.head(args.erp_count).iterrows():
            ts_code = row['ts_code']
            
            # 【修复】增加重试机制，解决网络抖动导致的个股缺失
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"[{idx+1}/{min(args.erp_count, len(results))}] 计算 {ts_code} ({row['name']})...")
                    erp_result = erp_calculator.calculate_erp(ts_code)
                    
                    if erp_result:
                        erp_data[ts_code] = {
                            'name': row['name'],
                            'industry': row['industry'] if pd.notna(row['industry']) else '',
                            **erp_result
                        }
                    # 成功后跳出重试循环
                    break 
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  ⚠ 计算出错，正在重试 ({attempt+1}/{max_retries})...")
                        time.sleep(2) # 失败后歇2秒
                    else:
                        print(f"  ✗ 最终失败: {e}")
            
            # 【修复】礼貌性暂停，防止被Tushare风控拦截
            time.sleep(0.1)
    
    save_results(results, erp_data, args.output, {
        'mode': mode_name,
        'min_dividend_yield': div_yield,
        'min_market_cap': mkt_cap,
        'min_price': min_price
    })
    
    print("\n✓ 全部完成! 请将 data 文件夹上传到 GitHub。")

if __name__ == '__main__':
    main()