from datetime import datetime

# ---------- helpers ----------
def fmt_pct(x, digits=2):
    if x is None: return "N/A"
    return f"{x*100:.{digits}f}%"

def fmt_pct_raw(x, digits=2):
    if x is None: return "N/A"
    return f"{x:.{digits}f}%"

def fmt_num(x):
    if x is None: return "N/A"
    absx = abs(x)
    if absx >= 1e12: return f"{x/1e12:.2f}T"
    if absx >= 1e9:  return f"{x/1e9:.2f}B"
    if absx >= 1e6:  return f"{x/1e6:.2f}M"
    if absx >= 1e3:  return f"{x/1e3:.2f}K"
    return f"{x:.2f}"

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur: 
            return default
        cur = cur[k]
    return cur

def format_date(dt):
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d")
    return str(dt)


FUND_KEY_RENAME = {
    # income statement (select)
    "revenues": "revenue",
    "cost_of_revenue": "cost of revenue",
    "gross_profit": "gross profit",
    "operating_expenses": "operating expenses",
    "sga": "SG&A",
    "operating_income": "operating income",
    "pretax_income": "pre-tax income",
    "income_tax_expense": "income tax expense",
    "interest_expense_oper": "interest expense",
    "net_income": "net income",
    # per-share
    "eps_basic": "EPS (basic)",
    "eps_diluted": "EPS (diluted)",
    "shares_basic": "basic shares",
    "shares_diluted": "diluted shares",
    # balance sheet (select)
    "total_assets": "total assets",
    "total_liabilities": "total liabilities",
    "equity": "shareholders' equity",
    "current_assets": "current assets",
    "current_liabilities": "current liabilities",
    "inventory": "inventory",
    "accounts_payable": "accounts payable",
    # cash flow (select)
    "cfo": "cash from operations",
    "cfi": "cash from investing",
    "cff": "cash from financing",
    "net_cash_flow": "net cash flow",
    # ratios (select)
    "gross_margin": "gross margin",
    "operating_margin": "operating margin",
    "net_margin": "net margin",
    "ocf_margin": "operating cash flow margin",
    "current_ratio": "current ratio",
    "quick_ratio": "quick ratio",
    "working_capital": "working capital",
    "liabilities_to_equity": "liabilities-to-equity",
    "debt_to_assets": "debt-to-assets",
    "interest_coverage": "interest coverage",
    "effective_tax_rate": "effective tax rate",
    "asset_turnover": "asset turnover",
    "sga_ratio": "SG&A ratio",
    "investing_cf_to_sales": "investing CF to sales",
    "financing_cf_to_sales": "financing CF to sales",
}

MACRO_KEY_RENAME = {
    "yield_1_month": "US 1M Treasury yield",
    "yield_3_month": "US 3M Treasury yield",
    "yield_2_year": "US 2Y Treasury yield",
    "yield_10_year": "US 10Y Treasury yield",
    "yield_30_year": "US 30Y Treasury yield",
    "cpi": "CPI (headline)",
    "cpi_core": "CPI (core)",
    "pce_core": "PCE price index (core)",
    "pce_spending": "PCE personal consumption (nominal)",
    "market_5_year": "Market 5Y breakeven (inflation)",
    "market_10_year": "Market 10Y breakeven (inflation)",
}

FUND_RATIO_PCT = {
    "gross_margin", "operating_margin", "net_margin", "ocf_margin",
    "effective_tax_rate", "sga_ratio", "investing_cf_to_sales", "financing_cf_to_sales"
}
TREND_RATIO_PCT = {
    "3d_return", "5d_return", "10d_return", "total_range_pct_vs_start",
    "max_close_ret_vs_start", "min_close_ret_vs_start"
}
VOL_ANNUALIZED = {"realized_vol_ann", "downside_vol_ann"}
VOL_RAW = {"realized_vol_raw_10d"}

def _render_macro(macro: dict) -> str:
    if not macro: return "Macro Snapshot: N/A"
    lines = ["Macro Snapshot:"]
    for k, v in macro.items():
        label = MACRO_KEY_RENAME.get(k, k)
        if "yield" in k or "market_" in k:
            lines.append(f"- {label}: {fmt_pct(v/100) if v and v>1 else fmt_pct(v)}")
        elif k in {"cpi", "cpi_core", "pce_core"}:
            lines.append(f"- {label}: {fmt_num(v)}")
        elif k == "pce_spending":
            lines.append(f"- {label}: {fmt_num(v)}")
        else:
            lines.append(f"- {label}: {fmt_num(v)}")
    return "\n".join(lines)

def _render_fundamentals(fund: dict) -> str:
    if not fund: return "Fundamentals: N/A"
    fin = fund.get("financials", {}) or {}
    pick_order = [
        # scale & profitability
        "revenues","gross_profit","operating_income","net_income",
        # margins
        "gross_margin","operating_margin","net_margin","ocf_margin",
        # balance sheet quality
        "total_assets","total_liabilities","equity","current_ratio","quick_ratio",
        "liabilities_to_equity","debt_to_assets","working_capital",
        # efficiency & coverage
        "asset_turnover","interest_coverage","effective_tax_rate",
        # cash flow view
        "cfo","cfi","cff","net_cash_flow",
    ]
    lines = [f"Fundamentals ({fund.get('fiscal_period','')} {fund.get('fiscal_year','')}):"]
    for k in pick_order:
        if k not in fin: 
            continue
        label = FUND_KEY_RENAME.get(k, k)
        val = fin[k]
        if k in FUND_RATIO_PCT:
            lines.append(f"- {label}: {fmt_pct(val)}")
        elif k in {"debt_to_assets","liabilities_to_equity","asset_turnover","current_ratio","quick_ratio","interest_coverage"}:
            lines.append(f"- {label}: {val:.2f}" if isinstance(val,(int,float)) else f"- {label}: {val}")
        else:
            lines.append(f"- {label}: {fmt_num(val)}")
    return "\n".join(lines)

def _render_news(sample: dict) -> str:
    news = sample.get("stock_data", {}).get("input", {}).get("News", [])
    if not news:
        return "N/A"
    
    lines = []
    for n in news:
        date = n.get("date", "Unknown Date")
        desc = n.get("description", "No description available")
        sentiment = n.get("sentiment")
        reason = n.get("sentiment_reason")

        if sentiment is not None and reason is not None:
            lines.append(f"- {date}: {desc} | Sentiment: {sentiment} | Sentiment Reason: {reason}")
        elif sentiment is not None:
            lines.append(f"- {date}: {desc} | Sentiment: {sentiment}")
        elif reason is not None:
            lines.append(f"- {date}: {desc} | Sentiment Reason: {reason}")
        else:
            lines.append(f"- {date}: {desc}")
    
    return "\n".join(lines)


def build_llm_prompt(sample: dict, task="sft_new") -> str:
    # Basic header
    ticker = safe_get(sample, "basic_info", "ticker") or sample.get("ticker", "N/A")
    name = safe_get(sample, "basic_info", "name") or "N/A"
    date_str = format_date(sample.get("date", "N/A"))
    desc = safe_get(sample, "basic_info", "description") or "N/A"

    macro = sample.get("macro_data", {}) or {}
    fund = sample.get("fundamental_data", {}) or {}

    if task == "sftv1":
        system_block = (
            "You are an expert equity analyst specializing in short-horizon (3–10 trading days) stock behavior. "
            "You will read the the time series soft tokens and the context."
            "Your task is to answer the question."
        )
    elif task == "sft_new":
        from dataloader.prompt import SYSTEM_PROMPT_SFT_NEW
        system_block = SYSTEM_PROMPT_SFT_NEW
    else:
        raise ValueError("task must be 'sftv1' or 'sft_new'")
    

    header = (
        f"Context Date: {date_str}\n"
        f"Company: {name} ({ticker.upper()})\n"
        f"Business Summary: {desc}"
    )

    macro_txt = _render_macro(macro)
    fund_txt = _render_fundamentals(fund)
    # primer = _finance_primer()
    news_txt = _render_news(sample)


    text = f"""{system_block}
        Today is {sample['date_obj'].strftime("%Y-%m-%d")}. 
        The basic information of the stock '{sample['ticker']}': {sample['basic_info']['description']}. 
        Today's global macro data: {macro_txt}. 
        The company's most recent financial fundamentals: {fund_txt}. 
        The news and potential sentiments: {news_txt[:2000] + "..." if len(news_txt) > 2000 else news_txt}. 
        The past 90 days' closing prices: {sample['stock_data']['input']['Prices']['Close'][:]}."""
        
    return text, system_block, header, macro_txt, fund_txt, news_txt

def batch_build_prompts(samples, task="sft_new"):
    if isinstance(samples, dict):
        return [build_llm_prompt(samples, task=task)[0]]
    return [build_llm_prompt(s, task=task)[0] for s in samples]






    # task = (
    #     "Task:\n"
    #     "1) Assess the short-term (3–10 trading days) directional bias (up/flat/down) and confidence.\n"
    #     "2) Identify key drivers using the metrics (volatility, drawdown/run-up, trend buckets, volume profile, macro, margins/liquidity, etc.).\n"
    #     "3) State upside/downside scenarios with approximate probabilities.\n"
    #     "4) List 2–4 crisp monitoring signals (e.g., volume vs day0, range expansion, breach of recent max close, macro yield shifts).\n"
    #     "Output format: A short paragraph for bias, a bullet list for drivers, a bullet list for scenarios with probabilities, and a bullet list for monitoring."
    # )
