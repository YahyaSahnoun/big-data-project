FROM apache/hive:4.0.0

COPY postgresql-42.7.5.jar /opt/hive/lib/

ENV DB_DRIVER=postgres
ENV SERVICE_NAME=metastore
ENV IS_RESUME=true
