from src.processing import DataProcessor

query = '''
SELECT
    user_id,
    device_type_canonical,
    order_id,
    created_dt_msk AS order_dt,
    created_ts_msk AS order_ts,
    currency_code,
    revenue,
    tickets_count,
    created_dt_msk::date - LAG(created_dt_msk::date) OVER (
        PARTITION BY user_id
        ORDER BY created_dt_msk
    ) AS days_since_prev,
    p.event_id,
    e.event_name_code AS event_name,
    e.event_type_main,
    p.service_name,
    r.region_name,
    c.city_name
FROM afisha.purchases AS p
INNER JOIN afisha.events AS e ON e.event_id = p.event_id
INNER JOIN afisha.city AS c ON c.city_id = e.city_id
INNER JOIN afisha.regions AS r ON r.region_id = c.region_id
WHERE device_type_canonical IN ('mobile', 'desktop') AND e.event_type_main != 'фильм'
ORDER BY user_id ASC;
'''
def main():
    afisha = DataProcessor()

    df = afisha.load_data(query)
    df_currency = afisha.currency_load()
    df = afisha.data_preparation(df, df_currency)
    users_profile = afisha.users_profile_creation(df)
    afisha.retention_count(users_profile)
    afisha.correlation_analysis(users_profile)

if __name__ == "__main__":
    main()


