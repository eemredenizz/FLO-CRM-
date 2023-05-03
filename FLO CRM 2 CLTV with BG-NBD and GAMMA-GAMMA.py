##############################################################################
# CRM PROJESİ 2 // FLO ŞİRKETİ İÇİN BG-NBD ve GAMMA-GAMMA İLE  CLTV TAHMİNİ
# (CRM PROJECT 2 // CLTV PREDICTION WITH BG-NBD and GAMMA-GAMMA FOR FLO COMPANY)
##############################################################################

##################################
# preparing data (veriyi hazırlama)
##################################

# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = (quartile3 + 1.5 * interquantile_range).__int__()
    low_limit = (quartile1 - 1.5 * interquantile_range).__int__()
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

df["order_num_total_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_ever"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

df.dtypes

df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

########################################################
# 2- Creation Of CLTV data structure (CLTV Veri Yapısının Oluşması)
########################################################
# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç

df.sort_values("last_order_date", ascending=False)

today_date = dt.datetime(2021,6,2)

df["recency"] = df["last_order_date"] - df["first_order_date"]

cltv = df.groupby("master_id").agg({"recency": lambda recency: recency,
                                    "first_order_date": lambda first_order_date: (today_date - first_order_date),
                                    "order_num_total_ever": lambda order_num_total_ever: order_num_total_ever.sum(),
                                    "customer_value_total_ever": lambda customer_value_total_ever: customer_value_total_ever.sum()})

cltv.columns = ["recency_cltv_weekly", "T_weekly", "frequency", "monetary_cltv_avg"]

cltv["recency_cltv_weekly"] = cltv["recency_cltv_weekly"] / 7
cltv["T_weekly"] = cltv["T_weekly"] / 7
cltv["monetary_cltv_avg"] = cltv["monetary_cltv_avg"] / cltv["frequency"]
cltv["recency_cltv_weekly"] = cltv["recency_cltv_weekly"].dt.days
cltv["T_weekly"] = cltv["T_weekly"].dt.days


####################################################################
#3- Setting Up BG/NBD, GAMMA-GAMMA Models and Calculating CLTV (BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması)
####################################################################

bgf = bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv["frequency"],
        cltv["recency_cltv_weekly"],
        cltv["T_weekly"])

cltv["expec_sales_3_month"] = bgf.predict(4*3,
                                          cltv["frequency"],
                                          cltv["recency_cltv_weekly"],
                                          cltv["T_weekly"])

cltv["expec_sales_6_month"] = bgf.predict(4*6,
                                          cltv["frequency"],
                                          cltv["recency_cltv_weekly"],
                                          cltv["T_weekly"])

ggf = ggf = GammaGammaFitter(penalizer_coef=0.1)

ggf.fit(cltv["frequency"], cltv["monetary_cltv_avg"])

cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv["frequency"], cltv["monetary_cltv_avg"])

cltv["CLTV"] = ggf.customer_lifetime_value(bgf,
                                           cltv["frequency"],
                                           cltv["recency_cltv_weekly"],
                                           cltv["T_weekly"],
                                           cltv["monetary_cltv_avg"],
                                           time=6,
                                           freq="W",
                                           discount_rate=0.01)

cltv["segment"] = pd.qcut(cltv["CLTV"], 4, ["cant_loose", "about_to_sleep", "need_attention","promising"])


#################################################################
# Yöneticilere Tavsiyeler (Advices to executives)
#################################################################

# cant_loose segmentindeki müşterileri kaybetme ihtimaliniz yüksek. O gruptaki müşterilerle özel olarak iletişime geçebilir ve kişiye özel kampanyalar yapabilirsiniz.
# there is a big possibility you are gonna lost your customers who's at can't loose segment. You can get contact with them and you can make a personalized campaign.
# need_attention segmentindeki müşterileri iyi bir reklamla harekete geçirebilirsiniz. Sosyal medya ve televizyonda yayınlayacağınız bu reklamlarla hedef kitleye ulaşmanız mümkün.
# you can mobilize customers in the need_attention segment with a good advertisement. It is possible to reach the target customers with these advirtisements that you will publish on social media and television.