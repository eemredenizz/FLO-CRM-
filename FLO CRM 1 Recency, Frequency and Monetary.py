#################################################################
# CRM PROJESİ 1 // FLO MÜŞTERİ SEGMENTASYONU (CRM PROJECT 1 // FLO CUSTOMER SEGMANTATION)
#################################################################
#####################################
# 1- Veri Ön Hazırlama (Preparing Data)
#####################################
import pandas as pd
import numpy as np
import datetime as dt
pd.set_option("display.max_columns",None)
pd.set_option("display.float_format", lambda x: "%3.f" %x)

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()

df.head(10)
df.columns
df.describe().T
df.isnull().sum()

df.dtypes

df["customer_order_num_total_ever"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["customer_value_total_ever"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

df["customer_value_total_ever"].sum()
df["customer_order_num_total_ever"].sum()
df["master_id"].__len__()

df.sort_values("customer_value_total_ever",ascending=False).head(10)
df.sort_values("customer_order_num_total_ever",ascending=False).head(10)


def preparing_data(dataframe):
    import pandas as pd
    import numpy as np
    import datetime as dt
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: "%3.f" % x)
    df["customer_order_num_total_ever"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
    df["customer_value_total_ever"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    df["first_order_date"] = pd.to_datetime(df["first_order_date"])
    df["last_order_date"] = pd.to_datetime(df["last_order_date"])
    df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
    df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])


##############################################################
# 2- Recency, Frequency ve Monetary değerlerinin oluşturulması (Creating Recency, Frequency and Monetary values)
##############################################################
# Recency = today_date - Customers last order date
# Frequency = Customers total number of purchase
# Monetary = total amounts that paid by customers

today_date = dt.datetime(2021,6,2)

rfm = df.groupby("master_id").agg({"last_order_date": lambda last_order_date: (today_date - last_order_date),
                                    "customer_order_num_total_ever": lambda customer_order_num_total_ever: customer_order_num_total_ever,
                                    "customer_value_total_ever": lambda customer_value_total_ever: customer_value_total_ever.sum()})
rfm.columns =["recency", "frequency", "monetary"]
rfm.head()

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["monetary_Score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2 ,3, 4, 5])

rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

seg_map = {r'[1-2][1-2]': "hibernating",
           r'[1-2][3-4]': "at_risk",
           r'[1-2]5': "cant_loose",
           r'3[1-2]': "about_to_sleep",
           r'33': "need_attention",
           r'[3-4][4-5]':"loyal_customers",
           r'41': "promising",
           r'51': "new_customers",
           r'[4-5][2-3]': "potential_loyallist",
           r'5[4-5]': "champions"}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

rfm.groupby("segment").agg({"recency":lambda recency: recency.mean(),
                            "frequency":lambda frequency: frequency.mean(),
                            "monetary": lambda monetary: monetary.mean()})

# İş problemi 1
# FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
#tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
#iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
#yapan kişiler özel olarak iletişim kurulacak müşteriler.Bu müşterilerin id numaralarını csv dosyasına kaydediniz.

rfm_new = pd.merge(rfm, df, on="master_id")
woman_cat_loyal_or_champion_customers = rfm_new[(rfm_new["segment"] == "champions") | (rfm_new["segment"] == "loyal_customers") & (rfm_new["interested_in_categories_12"].str.contains("KADIN"))]

woman_cat_loyal_or_champion_customers["master_id"].to_csv("woman_cat_loyal_or_champion_customers_masterid.csv")


# İş problemi 2
# Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
# gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.

man_and_child_cat_cant_loose_or_about_to_sleep_customers = rfm_new[(rfm_new["segment"] == "cant_loose") | (rfm_new["segment"] == "about_to_sleep") & (rfm_new["interested_in_categories_12"].str.contains("ERKEK")) | (rfm_new["interested_in_categories_12"].str.contains("COCUK"))]

man_and_child_cat_cant_loose_or_about_to_sleep_customers["master_id"].to_csv("man_and_child_cat_cant_loose_or_about_to_sleep_customers_masterid.csv")
