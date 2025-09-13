import os, json, re, difflib, datetime
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import requests
import bcrypt
import smtplib
from email.message import EmailMessage
import streamlit as st
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# ==============================
# Load environment variables
# ==============================
load_dotenv()
GROQ_API_ENV = os.getenv("groq_api_key", "").strip()
ORS_API_KEY = os.getenv("ORS_API_KEY", "").strip()

# ==============================
# Utility: JSON persistence
# ==============================
DATA_DIR = Path(".")
USERS_FILE = DATA_DIR / "users.json"
RESTOCK_FILE = DATA_DIR / "restock_events.json"

def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

users_db = load_json(USERS_FILE, {})
restock_events = load_json(RESTOCK_FILE, [])

# ==============================
# Auth helpers
# ==============================
def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

def check_pw(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        return False

def register_user(username, password, role='owner'):
    if not username or not password:
        return False, "Username / password required."
    if username in users_db:
        return False, "Username exists."
    users_db[username] = {
        "password": hash_pw(password),
        "role": role,
        "profile": {"email":"","phone":"","address":"","company":"","notes":""}
    }
    save_json(USERS_FILE, users_db)
    return True, "Registered."

def update_profile(username, **fields):
    if username not in users_db:
        return False, "User missing."
    users_db[username]["profile"].update(fields)
    save_json(USERS_FILE, users_db)
    return True, "Profile updated."

def log_restock(username, product_name, city_name, qty, gap_before, gap_after):
    event = {
        "user": username,
        "product_name": product_name,
        "city_name": city_name,
        "qty": qty,
        "gap_before": gap_before,
        "gap_after": gap_after,
        "timestamp": pd.Timestamp.utcnow().isoformat()
    }
    restock_events.append(event)
    save_json(RESTOCK_FILE, restock_events)

def compute_performance(restock_events):
    if not restock_events:
        return pd.DataFrame(columns=['user','total_qty','actions','avg_fill_improve','speed_score','performance'])
    df = pd.DataFrame(restock_events)
    df['fill_improve'] = df['gap_before'] - df['gap_after']
    now = pd.Timestamp.utcnow()
    df['age_hours'] = (now - pd.to_datetime(df['timestamp']))/pd.Timedelta(hours=1)
    df['speed_score'] = 1/(1+df['age_hours'])
    agg = df.groupby('user').agg({
        'qty':'sum',
        'fill_improve':'mean',
        'speed_score':'sum',
        'product_name':'count'
    }).rename(columns={'product_name':'actions','fill_improve':'avg_fill_improve','qty':'total_qty'}).reset_index()
    agg['performance'] = (agg['avg_fill_improve'].clip(lower=0)) + 0.1*agg['speed_score'] + 0.01*agg['total_qty']
    return agg.sort_values('performance', ascending=False)

# ==============================
# ML: Demand & Urgency Model
# ==============================
@st.cache_data(show_spinner=False)
def train_demand_urgency_model(sales_df, stock_df):
    df = sales_df.copy()
    if 'date' not in df.columns:
        # fabricate a single recent date for all rows to still allow aggregation
        df['date'] = pd.Timestamp.today().normalize()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    lookback_days = 14
    recent = df[df['date'] >= df['date'].max() - timedelta(days=lookback_days-1)]
    if recent.empty:
        return None, None, pd.DataFrame()

    # Aggregate sales stats
    agg = (recent.groupby(['product_id','product_name','city_name'])
           .agg(units_sum=('units_sold','sum'),
                units_mean=('units_sold','mean'),
                units_std=('units_sold','std'),
                units_max=('units_sold','max'),
                units_min=('units_sold','min'))
           .reset_index())
    agg.rename(columns={
        'units_sum':'demand',
        'units_mean':'sales_rate',
        'units_std':'sales_std',
        'units_max':'max_sales',
        'units_min':'min_sales'
    }, inplace=True)

    merged = pd.merge(agg, stock_df, on=['product_id','product_name','city_name'], how='left')
    merged['stock_quantity'] = merged['stock_quantity'].fillna(0)
    merged['sales_std'] = merged['sales_std'].fillna(0)
    merged['stock_to_demand'] = merged['stock_quantity'] / (merged['demand'] + 1)
    merged['sales_volatility'] = (merged['max_sales'] - merged['min_sales']) / (merged['sales_rate'] + 1)

    feature_cols = ['demand','sales_rate','sales_std','max_sales','min_sales',
                    'stock_quantity','stock_to_demand','sales_volatility']
    X = merged[feature_cols].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    y = merged['demand']
    reg = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    reg.fit(Xs, y)
    merged['demand_pred'] = reg.predict(Xs)
    merged['urgency_pred'] = (merged['stock_quantity'] < 0.2 * merged['demand_pred']).astype(int)
    out_cols = ['product_id','product_name','city_name','demand','demand_pred','sales_rate','stock_quantity','urgency_pred']
    return reg, scaler, merged[out_cols].sort_values('urgency_pred', ascending=False)

# ==============================
# Chatbot context helpers
# ==============================
def get_recent_sales_summary(sales_df, top_n=5):
    if 'units_sold' not in sales_df.columns:
        return pd.DataFrame()
    grp_cols = [c for c in ['product_id','product_name'] if c in sales_df.columns]
    if not grp_cols:
        return pd.DataFrame()
    return (sales_df.groupby(grp_cols)['units_sold']
            .sum().reset_index()
            .sort_values('units_sold', ascending=False)
            .head(top_n))

@st.cache_data(show_spinner=False)
def build_compact_context(sales_df, ml_pred_df, top_n=5, urgent_n=5):
    top_products_df = get_recent_sales_summary(sales_df, top_n)
    top_products = top_products_df.to_dict(orient='records')
    urgent_rows = ml_pred_df[ml_pred_df['urgency_pred']==1] if not ml_pred_df.empty else pd.DataFrame()
    urgent_sample = (urgent_rows.sort_values('demand_pred', ascending=False)
                     .head(urgent_n)[['product_name','city_name','demand_pred','stock_quantity']]
                     .to_dict(orient='records'))
    context = {
        'summary': {
            'total_products': int(sales_df['product_id'].nunique()) if 'product_id' in sales_df.columns else 0,
            'total_cities': int(sales_df['city_name'].nunique()) if 'city_name' in sales_df.columns else 0,
            'urgent_count': int(len(urgent_rows))
        },
        'top_products': top_products,
        'urgent_items_sample': urgent_sample
    }
    return json.dumps(context, separators=(',',':'))

def find_product_context(user_query, ml_pred_df):
    q = user_query.lower()
    for r in ml_pred_df.itertuples():
        if str(r.product_name).lower() in q:
            return {
                'product_name': r.product_name,
                'city_name': r.city_name,
                'demand_pred': float(r.demand_pred),
                'stock_quantity': int(r.stock_quantity),
                'urgency_pred': int(r.urgency_pred)
            }
    return None

def parse_product_and_stock(user_query):
    stock_match = re.search(r'(\d{2,9})\s*(?:units|stock)', user_query, re.IGNORECASE)
    stock_val = int(stock_match.group(1)) if stock_match else None
    cleaned = re.sub(r'\d+',' ', user_query)
    cleaned = re.sub(r'\b(have|stock|units|allocate|need|needed|for|what|which|cities|where|is|are|i|of|with|how|much)\b',' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+',' ', cleaned).strip()
    return cleaned or None, stock_val

def best_match_product(fragment, ml_pred_df):
    if not fragment:
        return None
    names = ml_pred_df['product_name'].unique().tolist()
    scores = [(difflib.SequenceMatcher(None, fragment.lower(), n.lower()).ratio(), n) for n in names]
    scores.sort(reverse=True)
    return scores[0][1] if scores and scores[0][0] >= 0.4 else None

def query_chatbot(user_query, sales_df, stock_df, groq_api_key, ml_pred_df):
    if ml_pred_df.empty:
        return "No prediction data available yet."
    compact_context = build_compact_context(sales_df, ml_pred_df, top_n=3, urgent_n=5)
    parsed_name, provided_stock = parse_product_and_stock(user_query)
    product_rows = None
    if parsed_name:
        match_name = best_match_product(parsed_name, ml_pred_df)
        if match_name:
            product_rows = ml_pred_df[ml_pred_df['product_name']==match_name].copy()
    if product_rows is not None and not product_rows.empty:
        product_rows['gap'] = (product_rows['demand_pred'] - product_rows['stock_quantity']).clip(lower=0).round(2)
        if provided_stock and provided_stock > 0:
            total_gap = product_rows['gap'].sum()
            to_allocate = min(provided_stock, total_gap) if total_gap>0 else 0
            if to_allocate>0:
                raw = product_rows['gap']/total_gap*to_allocate
                alloc_int = np.floor(raw).astype(int)
                remainder = int(to_allocate - alloc_int.sum())
                if remainder>0:
                    frac = (raw - alloc_int).values
                    order = np.argsort(-frac)[:remainder]
                    alloc_int.iloc[order] += 1
                product_rows['allocation'] = alloc_int
                product_rows['post_gap'] = (product_rows['gap'] - product_rows['allocation']).clip(lower=0)
        prod_summary = product_rows[['city_name','demand_pred','stock_quantity','gap'] +
                                    (['allocation','post_gap'] if 'allocation' in product_rows.columns else [])] \
                       .sort_values('gap', ascending=False).head(15).to_dict(orient='records')
    else:
        prod_summary = None

    wrapper = {
        'context': compact_context,
        'focus_product_fragment': parsed_name,
        'supplied_stock': provided_stock,
        'focus_rows': prod_summary
    }

    prompt = (
        "You are a concise Q-Commerce allocation assistant.\n"
        "If focus_rows present: output a markdown table with columns in order: City|Demand|Stock|Gap"
        "|Allocation|PostGap (omit last two if absent). Keep numeric values unchanged.\n"
        "After table give one sentence recommendation (<25 words).\n"
        "JSON_DATA=" + json.dumps(wrapper, separators=(',',':')) +
        "\nQUESTION=" + user_query
    )
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role":"system","content":"You are a precise Q-Commerce analytics assistant."},
            {"role":"user","content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.25
    }
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers=headers, json=payload, timeout=25)
        if r.status_code==200:
            return r.json()['choices'][0]['message']['content'].strip()
        return f"API Error {r.status_code}: {r.text[:120]}"
    except Exception as e:
        return f"Request failed: {e}"

# ==============================
# Email notification
# ==============================
def send_notification_email(product_name, city_name, to_email,
                            smtp_host, smtp_port, smtp_user, smtp_pass):
    if not all([smtp_host, smtp_port, smtp_user, smtp_pass, to_email]):
        return False, "Missing SMTP credentials or recipient."
    try:
        msg = EmailMessage()
        msg['Subject'] = f"Restock Needed: {product_name}"
        msg['From'] = smtp_user
        msg['To'] = to_email
        msg.set_content(f"The product '{product_name}' is low or out of stock in {city_name}. Please refill.")
        with smtplib.SMTP(smtp_host, int(smtp_port), timeout=25) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, "Email sent."
    except Exception as e:
        return False, str(e)

# ==============================
# ORS Travel Time / Distance
# ==============================
@st.cache_data(show_spinner=False)
def get_travel_info(origin_lonlat, dest_lonlat, ors_key):
    if not ors_key:
        return None, None, "Missing ORS_API_KEY"
    url = "https://api.openrouteservice.org/v2/matrix/driving-car"
    body = {"locations":[origin_lonlat, dest_lonlat],"metrics":["distance","duration"]}
    try:
        resp = requests.post(url, json=body,
                             headers={"Authorization": ors_key,"Content-Type":"application/json"},
                             timeout=20)
        if resp.status_code==200:
            data = resp.json()
            duration_min = round(data['durations'][0][1]/60,2)
            distance_km = round(data['distances'][0][1]/1000,2)
            return duration_min, distance_km, None
        return None, None, f"ORS {resp.status_code}: {resp.text[:100]}"
    except Exception as e:
        return None, None, str(e)

# Basic city coordinate mapping (extend as needed)
CITY_COORDS = {
    "agra": [78.0081,27.1767],
    "ahmedabad":[72.5714,23.0225],
    "bangalore":[77.5946,12.9716],
    "bengaluru":[77.5946,12.9716],
    "mumbai":[72.8777,19.0760],
    "delhi":[77.2090,28.6139],
    "chennai":[80.2707,13.0827],
    "hyderabad":[78.4867,17.3850],
    "pune":[73.8567,18.5204],
    "kolkata":[88.3639,22.5726],
    "jaipur":[75.7873,26.9124],
    "noida ghaziabad":[77.4326,28.6692],
    "gurugram faridabad":[77.3160,28.4500],
    "rest of india":[78.9629,20.5937]
}

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="Q-Commerce Dashboard", layout="wide")
st.title("🤖 Q-Commerce Agent Dashboard")

# Sidebar: File upload + API key
st.sidebar.header("Data & Keys")
sales_file = st.sidebar.file_uploader("Upload sales.csv", type=["csv"])
stock_file = st.sidebar.file_uploader("Upload inventory.csv", type=["csv"])
sidebar_key = st.sidebar.text_input("Groq API Key", type="password")
groq_api_key = sidebar_key or GROQ_API_ENV
if not groq_api_key:
    st.sidebar.warning("Groq API key required for chatbot.")

# Auth panel
if 'auth_user' not in st.session_state: st.session_state.auth_user = None

with st.sidebar.expander("Authentication", expanded=False):
    if st.session_state.auth_user:
        st.success(f"Logged in as {st.session_state.auth_user}")
        if st.button("Logout"):
            st.session_state.auth_user = None
    else:
        st.markdown("**Login**")
        lu = st.text_input("Username", key="login_u")
        lp = st.text_input("Password", type="password", key="login_p")
        if st.button("Login"):
            if lu in users_db and check_pw(lp, users_db[lu]['password']):
                st.session_state.auth_user = lu
                st.experimental_rerun()
            else:
                st.error("Invalid credentials.")
        st.markdown("---")
        st.markdown("**Register**")
        ru = st.text_input("New Username", key="reg_u")
        rp = st.text_input("New Password", type="password", key="reg_p")
        role = st.selectbox("Role", ['owner','admin'], key="reg_role")
        if st.button("Register"):
            ok,msg = register_user(ru, rp, role)
            (st.success if ok else st.error)(msg)

current_user = st.session_state.auth_user

# Require data
if not (sales_file and stock_file):
    st.info("👆 Upload both sales.csv & inventory.csv to proceed.")
    st.stop()

# Load data
sales_df = pd.read_csv(sales_file)
stock_df = pd.read_csv(stock_file)

# Normalize expected column names
if 'stock' in stock_df.columns and 'stock_quantity' not in stock_df.columns:
    stock_df.rename(columns={'stock':'stock_quantity'}, inplace=True)
if 'units_sold' not in sales_df.columns:
    st.error("sales.csv must contain 'units_sold'.")
    st.stop()

for col in ['product_id','product_name','city_name']:
    if col in sales_df.columns:
        sales_df[col] = sales_df[col].astype(str).str.strip()
    if col in stock_df.columns:
        stock_df[col] = stock_df[col].astype(str).str.strip()

if 'product_id' in sales_df.columns:
    sales_df['product_id'] = sales_df['product_id'].astype(str)
if 'product_id' in stock_df.columns:
    stock_df['product_id'] = stock_df['product_id'].astype(str)

if not current_user:
    st.warning("Login to access analytics.")
    st.stop()

# Train model BEFORE tabs so all tabs can use ml_pred_df
reg, scaler, ml_pred_df = train_demand_urgency_model(sales_df, stock_df)
if ml_pred_df.empty:
    st.error("Not enough data to build demand model.")
    st.stop()

tabs = st.tabs(["Dashboard","Urgency & Demand","Logistics","Leaderboard","Profile","Chatbot","Notify"])

# ---------------- Dashboard ----------------
with tabs[0]:
    st.markdown("### 📈 Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Units Sold", int(sales_df['units_sold'].sum()))
    c2.metric("Distinct Products", sales_df['product_id'].nunique() if 'product_id' in sales_df.columns else 0)
    c3.metric("Cities", sales_df['city_name'].nunique() if 'city_name' in sales_df.columns else 0)
    urgent_cnt = int((ml_pred_df['urgency_pred']==1).sum())
    c4.metric("Urgent (Predicted)", urgent_cnt)

    st.markdown("#### Top Products (All-Time Units Sold)")
    top_df = get_recent_sales_summary(sales_df, top_n=5)
    st.dataframe(top_df, use_container_width=True, height=230)

    # Daily highlight builder
    st.markdown("### 🔁 Daily Highlights (Slack / Teams)")
    slack_wh = st.text_input("Slack Webhook URL", type='password')
    teams_wh = st.text_input("Teams Webhook URL", type='password')

    def build_highlights():
        today = datetime.date.today().isoformat()
        urgent_sample = ml_pred_df.sort_values('demand_pred', ascending=False).head(3)
        lines = [f"Daily Highlights {today}",
                 f"Urgent predicted: {urgent_cnt}",
                 "Top 3 Predicted Demand:"]
        for r in urgent_sample.itertuples():
            lines.append(f"- {r.product_name} ({r.city_name}): {round(r.demand_pred,1)} demand_pred stock {int(r.stock_quantity)}")
        perf_df = compute_performance(restock_events)
        if not perf_df.empty:
            lines.append("Top Restock Performers:")
            for r in perf_df.head(3).itertuples():
                lines.append(f"- {r.user} perf={round(r.performance,2)} fillΔ={round(r.avg_fill_improve,2)} acts={r.actions}")
        return "\n".join(lines)

    if st.button("Preview Highlights"):
        st.text(build_highlights())

    def post_webhook(url, text):
        try:
            r = requests.post(url, json={"text": text}, timeout=8)
            return r.status_code, r.text[:120]
        except Exception as e:
            return None, str(e)

    if st.button("Post Highlights", disabled=not (slack_wh or teams_wh)):
        msg = build_highlights()
        if slack_wh:
            code, resp = post_webhook(slack_wh, msg); st.write(f"Slack: {code} {resp}")
        if teams_wh:
            code, resp = post_webhook(teams_wh, msg); st.write(f"Teams: {code} {resp}")

# --------- Urgency & Demand -----------
with tabs[1]:
    st.subheader("📊 Predicted Demand & Urgency")
    urgent_only = st.checkbox("Show only urgent", value=False)
    view_df = ml_pred_df[ml_pred_df['urgency_pred']==1] if urgent_only else ml_pred_df
    st.dataframe(view_df, use_container_width=True, height=450)

# -------------- Logistics (NEW ORS) --------------
with tabs[2]:
    st.subheader("🚚 Logistics Travel Time (OpenRouteService)")
    if not ORS_API_KEY:
        st.warning("Add ORS_API_KEY to .env for travel calculations.")
    all_cities = sorted(ml_pred_df['city_name'].str.lower().unique())
    origin_city = st.selectbox("Origin City (e.g. warehouse)", all_cities, index=0)
    dest_cities = st.multiselect("Destination Cities", all_cities, default=all_cities[:5])
    if st.button("Compute Travel Matrix", disabled=not ORS_API_KEY):
        rows = []
        for dc in dest_cities:
            if dc == origin_city: continue
            o_coord = CITY_COORDS.get(origin_city.lower())
            d_coord = CITY_COORDS.get(dc.lower())
            if not (o_coord and d_coord):
                rows.append({'destination': dc, 'time_min': None, 'distance_km': None, 'status':'coords missing'})
                continue
            t,d,err = get_travel_info(o_coord, d_coord, ORS_API_KEY)
            rows.append({'destination': dc,
                         'time_min': t,
                         'distance_km': d,
                         'status': 'ok' if not err else err})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("#### Add Travel Metrics to Urgent Products (Optional)")
    selected_product = st.selectbox("Select product for urgent logistics", ml_pred_df['product_name'].unique())
    subset = ml_pred_df[(ml_pred_df['product_name']==selected_product) & (ml_pred_df['urgency_pred']==1)]
    if ORS_API_KEY and not subset.empty and st.button("Compute Product Travel"):
        o_coord = CITY_COORDS.get(origin_city.lower())
        enriched = []
        for r in subset.itertuples():
            d_coord = CITY_COORDS.get(r.city_name.lower())
            if not (o_coord and d_coord):
                enriched.append({'city': r.city_name, 'demand_pred': r.demand_pred,
                                 'stock_qty': r.stock_quantity, 'time_min': None, 'distance_km': None})
            else:
                t,d,err = get_travel_info(o_coord, d_coord, ORS_API_KEY)
                enriched.append({'city': r.city_name, 'demand_pred': r.demand_pred,
                                 'stock_qty': r.stock_quantity,
                                 'time_min': t, 'distance_km': d})
        st.dataframe(pd.DataFrame(enriched).sort_values('demand_pred', ascending=False), use_container_width=True)

# -------------- Leaderboard --------------
with tabs[3]:
    st.subheader("🏆 Restock Performance Leaderboard")
    perf_df = compute_performance(restock_events)
    if perf_df.empty:
        st.info("No restock events logged yet.")
    else:
        st.dataframe(perf_df, use_container_width=True, height=400)

# -------------- Profile --------------
with tabs[4]:
    st.subheader("👤 Profile")
    prof = users_db.get(current_user, {}).get('profile', {})
    with st.form("profile_form"):
        email = st.text_input("Email", value=prof.get('email',''))
        phone = st.text_input("Phone", value=prof.get('phone',''))
        address = st.text_area("Address", value=prof.get('address',''))
        company = st.text_input("Company", value=prof.get('company',''))
        notes = st.text_area("Notes", value=prof.get('notes',''))
        submitted = st.form_submit_button("Save")
        if submitted:
            ok,msg = update_profile(current_user, email=email, phone=phone,
                                    address=address, company=company, notes=notes)
            (st.success if ok else st.error)(msg)
    st.write("Role:", users_db.get(current_user, {}).get('role','-'))

# -------------- Chatbot --------------
with tabs[5]:
    st.subheader("💬 Chatbot")
    q = st.text_input("Ask about demand, allocation, urgency, logistics...")
    if q:
        if not groq_api_key:
            st.error("Enter Groq API Key in sidebar.")
        else:
            with st.spinner("Thinking..."):
                answer = query_chatbot(q, sales_df, stock_df, groq_api_key, ml_pred_df)
            st.write("🤖 Agent:", answer)

# -------------- Notify --------------
with tabs[6]:
    st.subheader("✉️ Notify Store (Urgent Items)")
    urgent_rows = ml_pred_df[ml_pred_df['urgency_pred']==1]
    if urgent_rows.empty:
        st.info("No urgent predicted rows.")
    else:
        urgent_rows = urgent_rows.reset_index(drop=True)
        options = [
            f"{r.product_name} | {r.city_name} | DemandPred={int(r.demand_pred)} | Stock={int(r.stock_quantity)}"
            for r in urgent_rows.itertuples()
        ]
        sel = st.selectbox("Select urgent item", options)
        sel_row = urgent_rows.iloc[options.index(sel)] if sel else None
        to_email = st.text_input("Store Email")
        with st.expander("SMTP Settings"):
            smtp_host = st.text_input("SMTP Host", value=os.getenv('SMTP_HOST',''))
            smtp_port = st.text_input("SMTP Port", value=os.getenv('SMTP_PORT','587'))
            smtp_user = st.text_input("SMTP User", value=os.getenv('SMTP_USER',''))
            smtp_pass = st.text_input("SMTP Password", type='password', value=os.getenv('SMTP_PASS',''))
        if st.button("Notify Store", disabled=not (to_email and sel_row is not None)):
            gap_before = max(sel_row.demand_pred - sel_row.stock_quantity, 0)
            with st.spinner("Sending..."):
                ok,msg = send_notification_email(sel_row.product_name, sel_row.city_name,
                                                 to_email, smtp_host, smtp_port, smtp_user, smtp_pass)
            if ok:
                st.success(msg)
                # simulate restock log (example qty = gap_before)
                log_restock(current_user, sel_row.product_name, sel_row.city_name,
                            int(gap_before), gap_before, max(gap_before - gap_before,0))
            else:
                st.error(f"Failed: {msg}")
