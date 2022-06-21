import pygeohash as gh
from pyspark.sql.functions import regexp_extract


def reg_lat_lon(option):
    if option == "latitude":
        return "^(\+|-|)?(?:90(?:(?:\.0{1,10})?)|(?:[0-9]|[1-8][0-9])(?:(?:\.[0-9]{1,})?))$"
    elif option == "longitude":
        return "^(\+|-)?(?:180(?:(?:\.0{1,10})?)|(?:[0-9]|[1-9][0-9]|1[0-7][0-9])(?:(?:\.[0-9]{1,10})?))$"


def conv_str_plus(col):
    if col < 0:
        pass
    else:
        col = "+"+str(col)
    return col
f_conv_str_plus = F.udf(conv_str_plus,T.StringType())


def geo_to_latlong(x,option):
    
    if x is not None:
        
        if option == 0:
            try:
                return [float(a) for a in gh.decode(x)][option]
            except:
                pass
        elif option == 1:
            try:
                return [float(a) for a in gh.decode(x)][option]
            except:
                pass
        
        else:
            return None
    
    else:
        
        return None

f_geo_to_latlong = F.udf(geo_to_latlong,T.FloatType())


def ll_gh_cols(df,max_records=10000):
    
    lat_cols,long_cols,gh_cols = [],[],[]
    for i in df.dtypes:
        if i[1] in ('float','double','float32','float64'):
            c=0
            for j in [lat_reg,long_reg]:
                if c == 0:
                    x = df.withColumn("_",regexp_extract(f_conv_str_plus(i[0]),j,0))
                    max_val = abs(float(x.agg(F.max(i[0])).rdd.flatMap(lambda x:x).collect()[0]))
                    if (x.groupBy("_").count().count() > 2) & (max_val <= 90):
                        lat_cols.append(i[0])
                        c=c+1

                    elif (x.groupBy("_").count().count() > 2) & (max_val > 90):
                        long_cols.append(i[0])            
                        c=c+1
            
        if i[1] in ('string','object'):
            x = df.select(F.col(i[0])).dropna().limit(max_records).withColumn("len_gh",F.length(F.col(i[0])))
            x_ = x.agg(F.max("len_gh")).rdd.flatMap(lambda x: x).collect()[0]
            if x_ <12:
                if x.withColumn("_",f_geo_to_latlong(i[0],F.lit(0))).groupBy("_").count().count() > 2:
                    gh_cols.append(i[0])
                else:
                    pass
            
    return lat_cols,long_cols,gh_cols
