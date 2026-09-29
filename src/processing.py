import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import phik
import os
from sqlalchemy import create_engine, URL, text
from dotenv import load_dotenv

load_dotenv('.env')

class DataProcessor:
    def __init__(self):
        connection_url = URL.create(
            drivername='postgresql+psycopg2',
            username=os.getenv('DB_USER'),
            password=os.getenv('DB_PWD'),
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT')),
            database=os.getenv('DB_NAME')
        )
        self.engine = create_engine(connection_url)
    
    def load_data(self, query):
        df = pd.read_sql_query(query, con=self.engine)
        return df

    def currency_load(self):
        df_currency = pd.read_csv('data/final_tickets_tenge_df.csv')
        df_currency['data'] = pd.to_datetime(df_currency['data'])
        return df_currency

    def data_preparation(self, df, df_currency):
        df = df.merge(
        df_currency[['data', 'curs']],
        left_on='order_dt',
        right_on='data',
        how='left',
        validate='many_to_one'
        ) # Объединили основную таблицу и таблицу с курсом валют

        df['revenue_rub'] = df['revenue']
        mask = df['currency_code'] == 'kzt'
        df.loc[mask, 'revenue_rub'] = (df.loc[mask,'revenue'] * df.loc[mask, 'curs'] / 100)
        # Добавили столбец revenue_rub, где все продажи посчитаны в рублях по курсу ЦБ
         
        df = df.drop_duplicates(
                subset=['user_id', 'order_ts', 'device_type_canonical', 'event_id', 'service_name', 'revenue'],
                keep='first'
                )
        # Удалили дубликаты, совпадающие по указанным столбцам

        df = df.loc[
                    (df['revenue_rub'] <= df['revenue_rub'].quantile(0.99)) &
                    (df['revenue_rub'] > 0)
                    ]
        # Отфильтровали выбросы - продажи с отрицательной выручкой и по квантилю 99
        columns = ['revenue', 'revenue_rub', 'days_since_prev']
        df[columns] = df[columns].apply(
                pd.to_numeric,
                downcast='float'
                )
        df['tickets_count'] = pd.to_numeric(
            df['tickets_count'],
            downcast = 'integer'
                )
        # Привели типы данных
        return df

    def users_profile_creation(self, df):
        df = df.sort_values(by='order_dt', ascending=True) 
        #создаем таблицу users_profile на основе группировки заказов по пользователям 
        users_profile = (df.sort_values(['order_ts', 'order_id'])
                .drop_duplicates(subset='user_id', keep='first')
            [['user_id',
            'order_dt',
            'device_type_canonical',
            'region_name', 
            'event_type_main'
            ]]
            .rename(columns={
            'device_type_canonical':'device_of_1_order',
            'order_dt': '1_order_date',
            'region_name': 'region_of_1_order',
            'event_type_main': '1_event'
            })
        )

        last_dates = df.groupby('user_id')['order_dt'].max()
        users_profile['last_order_date'] = (
            users_profile['user_id'].map(last_dates)
            )
        # Добавляем столбец last_order_date с датой последнего заказа

        users_statistics = (
            df.groupby('user_id', as_index=False)
            .agg(
                total_orders = ('order_id', 'nunique'),
                avg_revenue = ('revenue_rub','mean'),
                avg_days_between_orders = ('days_since_prev','mean')       
            )
        )
       # Добавляем столбцы с количеством заказов и  средним временем между ними
        users_profile = users_profile.merge(
            users_statistics,
            on='user_id',
            how='left'
            )
        
        # Добавляем столбец is_two. В нем true/false значит, были ли повторные заказы
        users_profile['is_two'] = users_profile['total_orders'] > 1
        return users_profile

    def retention_count(self, users_profile):
        events = (users_profile.groupby('1_event')
        .agg(
             repeat_share=('is_two', 'mean')
        ).sort_values('repeat_share', ascending=False)
        )

        device_of_1_order = (users_profile.groupby('device_of_1_order')
                    .agg(
                        repeat_share=('is_two','mean')
                    ).sort_values('repeat_share', ascending=False)
                    )

        regions = (users_profile.groupby('region_of_1_order')
        .agg(
              repeat_share=('is_two','mean'),
              total_users=('user_id','nunique')
        ))

        top_regions = regions.nlargest(10, 'total_users')
        top_regions = top_regions.sort_values('repeat_share',ascending=False)
        y = round(users_profile['is_two'].mean(), 3)

        def plot_retention(repeat_share, title, xlabel, ylabel):
            ax = repeat_share.plot(
            kind='bar',
            figsize=(10,5),
            color='skyblue',
            edgecolor='black',
            rot=45
            )
            ax.set(
                title=title,
                xlabel=xlabel,
                ylabel=ylabel
            )
            ax.axhline(
                y=y,
                color='red',
                linestyle='--',
                label=f'Средняя доля вернувшихся покупателей {y}'
            )
            ax.legend()
            plt.tight_layout()
            plt.show()
            return ax

        plot_retention(repeat_share=events['repeat_share'],
                       title='Доля пользователей по типу мероприятия, совершивших повторный заказ',
                       xlabel='тип мероприятия',
                       ylabel='Доля пользователей, купивших повторный билет'
        )

        plot_retention(repeat_share= top_regions['repeat_share'],
                       title='Доля пользователей по регионам, совершивших повторный заказ',
                        xlabel='Название региона',
                        ylabel='доля пользователей'
        )
        
        plot_retention(repeat_share= device_of_1_order['repeat_share'],
                    title='Доля пользователей по типу устройства, совершивших повторный заказ',
                    xlabel='Тип устройства',
                    ylabel='Доля пользователей'
        )

    def correlation_analysis(self, users_profile):
        columns = ['total_orders', 'device_of_1_order', 'region_of_1_order', '1_event','avg_revenue','avg_days_between_orders']

        correlation_matrix = users_profile[columns].phik_matrix(
        interval_cols=['avg_revenue', 'avg_days_between_orders']
            )

        def correlation_heat_map(title, correlation_matrix):
            plt.figure(figsize=(10, 7))
            sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            vmin=0,
            vmax=1,
            linewidths=0.5
            )
            plt.title(title)
            plt.tight_layout()
            plt.show()

        correlation_heat_map(
        title='Корреляция phi_k для всех пользователей',
        correlation_matrix=correlation_matrix
        )

        users_profile['order_segments'] = pd.cut(
        users_profile['total_orders'],
        bins=[0, 1, 2, 4, float('inf')],
        labels=['1 заказ', '2 заказа', '3–4 заказа', '5 и более']
            )
        columns = [
            'order_segments',
            'device_of_1_order',
            'region_of_1_order',
            '1_event',
            'avg_revenue',
            'avg_days_between_orders']

        correlation_matrix = users_profile[columns].phik_matrix(
            interval_cols=['avg_revenue', 'avg_days_between_orders']
        )

        correlation_heat_map(
            title='Корреляция phi_k по сегментам',
            correlation_matrix=correlation_matrix
        )

    print('''В dataframe есть статистка о 281 879 продажах в течение 150 дней события. Билеты покупали 22 000 уникальных пользователя.
В целом пользователи распределены равномерно с немногочисленными выбросами по количеству заказов и среднему чеку. Мне кажется, около 
1% выборки это не аккаунты людей, а корпоративные аккаунты. Например, вряд ли обычный человек покупает билеты по 4 раза в неделю 
непрерывно в течение 3 месяцев. По ним слишком мало данных для анализа, но это точка роста. Например, таким клиентам можно 
предоставлять бухгалтерские документы и оплату с расчетного счета, что увеличит их удержание.
Нужно обратить внимание на спортивные мероприятия. Покупатели билетов на них реже возвращаются, и в целом спорт это 
небольшая часть всех продаж (Например, на концерты продано 13315 билета в отчищенной выборке, на стендап 3241, а на спорт всего 2469.

Матрица корреляции показа 2 связи. Одну очевидную - между регионом и средним чеком. В богатых регионах России средний чек явно выше. 
Вторая неочевидная - между регионом и типом первого мероприятия. Значит, в каких-то регионах Афиша плохо представлена в каких-то
типах мероприятий. Это поле для дальнейших исследований. Как итог - я бы посоветовал обратить внимание на 3 вещи:
1. Возможно, нужно добавить корпоративные аккаунты с функциями для юридических лиц.
2. Нужно больше заниматься спортивными мероприятиями. Это точка роста''')